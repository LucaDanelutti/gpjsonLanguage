package it.necst.gpjson;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.Optional;
import java.util.TreeMap;

@ExportLibrary(InteropLibrary.class)
public final class Namespace implements TruffleObject {

    private final TreeMap<String, Object> map = new TreeMap<>();

    private final String name;

    public Namespace(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name == null ? "<root>" : name;
    }

    private void addInternal(String newName, Object newElement) {
        if (newName == null || newName.isEmpty()) {
            throw new GpJSONInternalException("cannot add element with name '" + newName + "' in namespace '" + name + "'");
        }
        if (map.containsKey(newName)) {
            throw new GpJSONInternalException("'" + newName + "' already exists in namespace '" + name + "'");
        }
        map.put(newName, newElement);
    }

    public void addNamespace(Namespace namespace) {
        addInternal(namespace.name, namespace);
    }

    @CompilerDirectives.TruffleBoundary
    public Optional<Object> lookup(String... path) {
        if (path.length == 0) {
            return Optional.empty();
        }
        return lookup(0, path);
    }

    private Optional<Object> lookup(int pos, String[] path) {
        Object entry = map.get(path[pos]);
        if (entry == null) {
            return Optional.empty();
        }
        if (pos + 1 == path.length) {
            return Optional.of(entry);
        } else {
            return entry instanceof Namespace ? ((Namespace) entry).lookup(pos + 1, path) : Optional.empty();
        }
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    public boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return map.keySet().toArray(new String[0]);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public boolean isMemberReadable(String member) {
        return map.containsKey(member);
    }

    @ExportMessage
    @CompilerDirectives.TruffleBoundary
    public Object readMember(String member) throws UnknownIdentifierException {
        Object entry = map.get(member);
        if (entry == null) {
            throw UnknownIdentifierException.create(member);
        }
        return entry;
    }
}
