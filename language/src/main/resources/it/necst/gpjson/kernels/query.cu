// Should match with the value in LeveledBitmapsIndex
#define MAX_NUM_LEVELS 16

#define OPCODE_END 0x00
#define OPCODE_STORE_RESULT 0x01
#define OPCODE_MOVE_UP 0x02
#define OPCODE_MOVE_DOWN 0x03
#define OPCODE_MOVE_TO_KEY 0x04
#define OPCODE_MOVE_TO_INDEX 0x05
#define OPCODE_EXPRESSION_STRING_EQUALS 0x06
#define OPCODE_MOVE_TO_INDEX_REVERSE 0x07

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

__global__ void executeQuery(char *file, long n, long *newlineIndex, long newlineIndexSize, long *string_index, long *leveled_bitmaps_index, long leveled_bitmaps_index_size, long level_size, char *query, int numResults, long *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  long linesPerThread = (newlineIndexSize+stride-1) / stride;

  long start = index * linesPerThread;
  long end = start + linesPerThread;

  for (long fileIndex = start; fileIndex < end && fileIndex < newlineIndexSize; fileIndex += 1) {
    long newlineStart = newlineIndex[fileIndex];
    long newlineEnd = (fileIndex + 1 < newlineIndexSize) ? newlineIndex[fileIndex+1] : n;

    while(file[newlineEnd] != '}' && newlineEnd > newlineStart) {
      newlineEnd--;
    }
    
    while(file[newlineStart] != '{' && newlineStart < newlineEnd) {
      newlineStart++;
    }

    long lineIndex = newlineStart;
    assert(file[newlineStart] == '{');
    assert(file[newlineEnd] == '}');

    int currentLevel = 0;
    int queryPos = 0;
    char currentOpcode;
    long levelEnd[MAX_NUM_LEVELS];
    levelEnd[0] = newlineEnd;
    for (int j = 1; j < MAX_NUM_LEVELS; j++) {
      levelEnd[j] = -1;
    }

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
          // to find the start of the actual value
          while(file[fileIndex] == ' ' && fileIndex < levelEnd[currentLevel]) {
            fileIndex++;
          }

          int resultIndex = fileIndex*2*numResults + numResultsIndex*2;
          result[resultIndex] = lineIndex;
          result[resultIndex+1] = levelEnd[currentLevel];
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
              long index = leveled_bitmaps_index[level_size * (currentLevel - 1) + endCandidate / 64];
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
            lineIndex = findNextStructuralChar(leveled_bitmaps_index, levelEnd[currentLevel], lineIndex, currentLevel, level_size);
            assert(file[lineIndex] == ':' || file[lineIndex] == '}');
            if (file[lineIndex] == ':') {
              long stringEnd = -1;
              for (long endCandidate = lineIndex-1; endCandidate > newlineStart; endCandidate--) {
                if ((string_index[endCandidate / 64] & (1L << endCandidate % 64)) != 0) {
                  stringEnd = endCandidate;
                  break;
                }
              }
              long stringStart = stringEnd - keyLen;
              if (stringStart < newlineStart || file[stringStart] != '"') {
                lineIndex++;
                lineIndex = findNextStructuralChar(leveled_bitmaps_index, levelEnd[currentLevel], lineIndex, currentLevel, level_size);
                assert(file[lineIndex] == ',' || file[lineIndex] == '}');
                if (file[lineIndex] == '}')
                    goto nextLine;
                goto searchKey;
              }
              for (int i = 0; i < keyLen; i++) {
                if (key[i] != file[stringStart + i + 1]) {
                  lineIndex++;
                  lineIndex = findNextStructuralChar(leveled_bitmaps_index, levelEnd[currentLevel], lineIndex, currentLevel, level_size);
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

          if (file[lineIndex] == '[') {
            currIndex[currentLevel] = 0;

            searchIndex:
            lineIndex++;
            if (currIndex[currentLevel] < index) {
              lineIndex = findNextStructuralChar(leveled_bitmaps_index, levelEnd[currentLevel], lineIndex, currentLevel, level_size);
              assert(file[lineIndex] == ',' || file[lineIndex] == ']');
              if (file[lineIndex] == ',') {
                currIndex[currentLevel]++;
                goto searchIndex;
              } else {
                goto nextLine;
              }
            }
          } else {
            goto nextLine;
          }
          break;
        }
      }
    }
    nextLine: ;
  }
}
