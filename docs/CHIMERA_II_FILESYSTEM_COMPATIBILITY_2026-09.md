# Chimera II Universal Filesystem Compatibility

The wide-register CPU repository now tracks the filesystem capability model used by ChimeraIIOS. Filesystem support is separated from ISA execution and is classified as native read/write, native read-only, userspace compatibility, or emulator-only.

The target families include Linux/Unix ext4, XFS, Btrfs, ZFS, JFS/JFS2, UFS, System V, F2FS, NILFS2, EROFS, SquashFS; DOS/Windows FAT/NTFS; optical ISO-9660/UDF; Amiga OFS/FFS/CrossDOS; HP-UX VxFS; OpenVMS ODS-2/ODS-5; HFS/HFS+; QNX; ADFS; AFS; 9P; NFS/SMB; cluster/distributed filesystems; pseudo-filesystems; flash filesystems; BeFS and other historical formats.

OpenVMS ODS-2/ODS-5 and AmigaDOS are modeled as foreign semantic layers rather than falsely normalized into POSIX.

See the authoritative integration matrix in ChimeraIIOS: `include/chimera/filesystem_matrix.hpp`.
