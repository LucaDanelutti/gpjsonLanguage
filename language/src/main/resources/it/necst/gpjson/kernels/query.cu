// Should match with the value in leveledBitmapsIndex
#define MAX_NUM_LEVELS 16

#define OPCODE_END 0x00
#define OPCODE_STORE_RESULT 0x01
#define OPCODE_MOVE_UP 0x02
#define OPCODE_MOVE_DOWN 0x03
#define OPCODE_MOVE_TO_KEY 0x04
#define OPCODE_MOVE_TO_INDEX 0x05
#define OPCODE_MOVE_TO_INDEX_REVERSE 0x06
#define OPCODE_MARK_POSITION 0x07
#define OPCODE_RESET_POSITION 0x08
#define OPCODE_EXPRESSION_STRING_EQUALS 0x09

__device__ int findNextStructuralChar(long *extIndex, int levelEnd, int lineIndex, int currentLevel, int levelSize) {
  long index = extIndex[levelSize * currentLevel + lineIndex / 64];
  while (index == 0 && lineIndex < levelEnd) {
    lineIndex += 64 - (lineIndex % 64);
    index = extIndex[levelSize * currentLevel + lineIndex / 64];
  }
  bool isStructural = (index & (1L << lineIndex % 64)) != 0;
  while (!isStructural && lineIndex < levelEnd) {
    lineIndex++;
    index = extIndex[levelSize * currentLevel + lineIndex / 64];
    isStructural = (index & (1L << lineIndex % 64)) != 0;
  }
  return lineIndex;
}

__device__ int findPreviousStructuralChar(long *extIndex, int levelStart, int lineIndex, int currentLevel, int levelSize) {
  long index = extIndex[levelSize * currentLevel + lineIndex / 64];
  while (index == 0 && lineIndex > levelStart) {
    lineIndex -= 64 - (lineIndex % 64);
    index = extIndex[levelSize * currentLevel + lineIndex / 64];
  }
  bool isStructural = (index & (1L << lineIndex % 64)) != 0;
  while (!isStructural && lineIndex > levelStart) {
    lineIndex--;
    index = extIndex[levelSize * currentLevel + lineIndex / 64];
    isStructural = (index & (1L << lineIndex % 64)) != 0;
  }
  return lineIndex;
}

__global__ void f(char *file, int fileSize, long *newlineIndex, int newlineIndexSize, long *stringIndex, long *leveledBitmapsIndex, long levelSize, char *query, int numResults, long *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long linesPerThread = (newlineIndexSize+stride-1) / stride;

  long start = index * linesPerThread;
  long end = start + linesPerThread;

  // Initialization
  long startInit = start * 2 * numResults;
  long endInit = end * 2 * numResults;
  for (long i = startInit; i < endInit && i < numResults * 2 * newlineIndexSize; i += 1) {
    result[i] = -1;
  }

  for (long fileIndex = start; fileIndex < end && fileIndex < newlineIndexSize; fileIndex += 1) {
    long lineStart = newlineIndex[fileIndex];
    long lineEnd = (fileIndex + 1 < newlineIndexSize) ? newlineIndex[fileIndex+1] : fileSize;

    while(file[lineEnd] != '}' && lineEnd > lineStart) {
      lineEnd--;
    }

    while(file[lineStart] != '{' && lineStart < lineEnd) {
      lineStart++;
    }

    if (lineStart == lineEnd)
      continue;

    long lineIndex = lineStart;
    assert(file[lineStart] == '{');
    assert(file[lineEnd] == '}');

    int currentLevel = 0;
    int queryPos = 0;
    char currentOpcode;
    long levelEnd[MAX_NUM_LEVELS];
    levelEnd[0] = lineEnd;
    for (int j = 1; j < MAX_NUM_LEVELS; j++) {
      levelEnd[j] = -1;
    }

    int markedPos[MAX_NUM_LEVELS];
    int markedPosLevel[MAX_NUM_LEVELS];
    int markedPosIndex = -1;

    int numResultsIndex = 0;

    char *key;
    int keyLen;
    int index;
    int currIndex[MAX_NUM_LEVELS];
    for (int j = 0; j < MAX_NUM_LEVELS; j++) {
      currIndex[j] = -1;
    }

    while (true) {
      currentOpcode = query[queryPos++];

      switch (currentOpcode) {
        case OPCODE_END: {
          goto nextLine;
        }
        case OPCODE_STORE_RESULT: {
          assert(numResultsIndex < numResults);
          // If we are storing a result, we are not in a string, so we can safely skip all whitespace
          // to find the start and the end of the actual value
          long endStr = levelEnd[currentLevel];
          while(file[lineIndex] == ' ' && lineIndex < levelEnd[currentLevel]) {
            lineIndex++;
          }
          while(file[endStr-1] == ' ' && endStr > lineIndex) {
            endStr--;
          }

          int resultIndex = fileIndex*2*numResults + numResultsIndex*2;
          result[resultIndex] = lineIndex;
          result[resultIndex+1] = endStr;
          assert(result[resultIndex] <= result[resultIndex+1]);
          numResultsIndex++;
          break;
        }
        case OPCODE_MOVE_UP: {
          lineIndex = levelEnd[currentLevel];
          levelEnd[currentLevel] = -1;
          currentLevel--;
          break;
        }
        case OPCODE_MOVE_DOWN: {
          currentLevel++;
          // Now we need to find the end of this level, unless we already have one
          if (levelEnd[currentLevel] == -1) {
            for (long endCandidate = lineIndex + 1; endCandidate <= levelEnd[currentLevel - 1]; endCandidate += 1) {
              long index = leveledBitmapsIndex[levelSize * (currentLevel - 1) + endCandidate / 64];
              if (index == 0) {
                endCandidate += 64 - (endCandidate % 64) - 1;
                continue;
              }
              bool isStructural = (index & (1L << endCandidate % 64)) != 0;
              if (isStructural) {
                levelEnd[currentLevel] = endCandidate;
                break;
              }
            }
            assert(levelEnd[currentLevel] != -1);
            while (file[lineIndex] == ' ') {
              lineIndex++;
            }
            assert(file[lineIndex] == '{' || file[lineIndex] == '[' || file[lineIndex] == '"');
          }
          break;
        }
        case OPCODE_MOVE_TO_KEY: {
          keyLen = 0;

          int i = 0;
          int b;
          while (((b = query[queryPos++]) & 0x80) != 0) {
            keyLen |= (b & 0x7F) << i;
            i += 7;
            assert(i <= 35);
          }
          keyLen = keyLen | (b << i);

          key = query + queryPos;
          queryPos += keyLen;

          if (file[lineIndex] == '{') {
            searchKey:
            lineIndex++;
            lineIndex = findNextStructuralChar(leveledBitmapsIndex, levelEnd[currentLevel], lineIndex, currentLevel, levelSize);
            assert(file[lineIndex] == ':' || file[lineIndex] == '}');
            if (file[lineIndex] == ':') {
              long stringEnd = -1;
              for (long endCandidate = lineIndex-1; endCandidate > lineStart; endCandidate--) {
                if ((stringIndex[endCandidate / 64] & (1L << endCandidate % 64)) != 0) {
                  stringEnd = endCandidate;
                  break;
                }
              }
              long stringStart = stringEnd - keyLen;
              if (stringStart < lineStart || file[stringStart] != '"') {
                lineIndex++;
                lineIndex = findNextStructuralChar(leveledBitmapsIndex, levelEnd[currentLevel], lineIndex, currentLevel, levelSize);
                assert(file[lineIndex] == ',' || file[lineIndex] == '}');
                if (file[lineIndex] == '}')
                    goto nextLine;
                goto searchKey;
              }
              for (int i = 0; i < keyLen; i++) {
                if (key[i] != file[stringStart + i + 1]) {
                  lineIndex++;
                  lineIndex = findNextStructuralChar(leveledBitmapsIndex, levelEnd[currentLevel], lineIndex, currentLevel, levelSize);
                  assert(file[lineIndex] == ',' || file[lineIndex] == '}');
                  if (file[lineIndex] == '}')
                    goto nextLine;
                  goto searchKey;
                }
              }
            }
            lineIndex++;
          } else {
            goto nextLine;
          }
          break;
        }
        case OPCODE_MOVE_TO_INDEX: {
          index = 0;

          int i = 0;
          int b;
          while (((b = query[queryPos++]) & 0x80) != 0) {
            index |= (b & 0x7F) << i;
            i += 7;
            assert(i <= 35);
          }
          index = index | (b << i);

          if (file[lineIndex] == '[' || file[lineIndex] == ',' || file[lineIndex] == ']') {
            if (file[lineIndex] == '[')
              currIndex[currentLevel] = 0;
            else
              currIndex[currentLevel]++;

            searchIndex:
            if (currIndex[currentLevel] < index) {
              lineIndex++;
              lineIndex = findNextStructuralChar(leveledBitmapsIndex, levelEnd[currentLevel], lineIndex, currentLevel, levelSize);
              assert(file[lineIndex] == ',' || file[lineIndex] == ']');
              if (file[lineIndex] == ',') {
                currIndex[currentLevel]++;
                goto searchIndex;
              } else {
                goto nextLine;
              }
            } else if (currIndex[currentLevel] > index) {
              lineIndex--;
              lineIndex = findPreviousStructuralChar(leveledBitmapsIndex, 0, lineIndex, currentLevel, levelSize);
              assert(file[lineIndex] == ',' || file[lineIndex] == '[');
              currIndex[currentLevel]--;
              goto searchIndex;
            } else {
              lineIndex++;
            }
          } else {
            goto nextLine;
          }
          break;
        }

        case OPCODE_MOVE_TO_INDEX_REVERSE: {
          index = 0;

          int i = 0;
          int b;
          while (((b = query[queryPos++]) & 0x80) != 0) {
            index |= (b & 0x7F) << i;
            i += 7;
            assert(i <= 35);
          }
          index = index | (b << i);

          lineIndex = levelEnd[currentLevel]-1;
          while (file[lineIndex] == ' ') {
            lineIndex--;
          }
          if (file[lineIndex] == ']' || file[lineIndex] == ',' || file[fileIndex] == '[') {
            if (file[lineIndex] == ']')
              currIndex[currentLevel] = 0;
            else
              currIndex[currentLevel]++;

            searchIndexReverse:
            if (currIndex[currentLevel] < index) {
              lineIndex--;
              lineIndex = findPreviousStructuralChar(leveledBitmapsIndex, 0, lineIndex, currentLevel, levelSize);
              assert(file[lineIndex] == ',' || file[lineIndex] == '[');
              if (file[lineIndex] == ',' || currIndex[currentLevel]+1 == index) {
                currIndex[currentLevel]++;
                goto searchIndexReverse;
              } else {
                goto nextLine;
              }
            } else if (currIndex[currentLevel] > index) {
              lineIndex++;
              lineIndex = findNextStructuralChar(leveledBitmapsIndex, levelEnd[currentLevel], lineIndex, currentLevel, levelSize);
              assert(file[lineIndex] == ',' || file[lineIndex] == ']');
              currIndex[currentLevel]--;
              goto searchIndex;
            } else {
              lineIndex++;
            }
          } else {
            goto nextLine;
          }
          break;
        }
        case OPCODE_MARK_POSITION: {
          assert(markedPosIndex++ < MAX_NUM_LEVELS);
          markedPos[markedPosIndex] = lineIndex;
          markedPosLevel[markedPosIndex] = currentLevel;
          break;
        }
        case OPCODE_RESET_POSITION: {
          assert(markedPosIndex >= 0);
          lineIndex = markedPos[markedPosIndex];
          currentLevel = markedPosLevel[markedPosIndex];
          markedPosIndex--;
          for (int i=currentLevel+1; i<MAX_NUM_LEVELS; i++) {
            if (levelEnd[i] == -1)
              break;
            levelEnd[i] = -1;
          }
          break;
        }
        case OPCODE_EXPRESSION_STRING_EQUALS: {
          keyLen = 0;

          int i = 0;
          int b;
          while (((b = query[queryPos++]) & 0x80) != 0) {
            keyLen |= (b & 0x7F) << i;
            i += 7;
            assert(i <= 35);
          }
          keyLen = keyLen | (b << i);

          key = query + queryPos;
          queryPos += keyLen;

          while(file[lineIndex] == ' ' && lineIndex < levelEnd[currentLevel]) {
            lineIndex++;
          }
          long stringEnd = levelEnd[currentLevel] - 1;
          while(file[stringEnd] == ' ' && lineIndex < stringEnd) {
            stringEnd--;
          }
          assert(file[lineIndex] == '"' && file[stringEnd] == '"');

          long stringLength = stringEnd - lineIndex + 1;
          if (stringLength != keyLen) {
            goto nextLine;
          }

          for (long k = 0; k < keyLen; k++) {
            if (key[k] != file[lineIndex + k]) {
              goto nextLine;
            }
          }
          break;
        }
        default: {
          assert(false);
          break;
        }
      }
    }
    nextLine: ;
  }
}