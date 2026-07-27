# Archive safety

The official riscv64 archive contains 92 members. A metadata-only inspection
before extraction found:

- no absolute paths or `..` traversal;
- no escaping symbolic or hard links;
- no device nodes;
- no setuid or setgid files;
- no unsafe ownership or executable-install step.

Four normal shared-library symlinks carry archive mode `0777`; that is symlink
metadata, not target-file permissions. The archive was extracted into a new
empty versioned root and no bundled install script was executed.

Status: **pass**.
