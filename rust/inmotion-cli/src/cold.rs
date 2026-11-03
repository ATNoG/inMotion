//! Cold-page-cache preparation for `--cold` benchmarks.
//!
//! `posix_fadvise(POSIX_FADV_DONTNEED)` asks the kernel to drop the cached pages
//! of a file, so the next read comes from storage — which is what a router's
//! first load after a reboot actually does. Every regular file in the artifact
//! directory is advised, because the loader maps or reads all of them
//! (`meta.json`, the weights, and in the training layout the golden vectors).
//!
//! A failed advise is an error, not a warning: the record would otherwise claim
//! `"cold": true` for a load that was served from the page cache, which is the
//! one thing this flag exists to prevent.

use std::fs;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;

/// Drop every regular file in `dir` from the page cache.
///
/// Returns the number of files advised. An empty directory is an error: there is
/// nothing to evict, so the load that follows cannot be cold.
pub fn purge_dir(dir: &Path) -> Result<usize, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let is_file = entry
            .file_type()
            .map_err(|e| format!("{}: {e}", entry.path().display()))?
            .is_file();
        if is_file {
            paths.push(entry.path());
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "{}: no files to evict from the page cache",
            dir.display()
        ));
    }
    for path in &paths {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let fd = file.as_raw_fd();
        // SAFETY: `fd` stays open for the call (owned by `file`), and the advice
        // arguments are a whole-file range on a valid descriptor.
        let rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        if rc != 0 {
            return Err(format!(
                "{}: posix_fadvise(DONTNEED): {} (errno {rc})",
                path.display(),
                io::Error::from_raw_os_error(rc)
            ));
        }
    }
    Ok(paths.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "inmotion-cold-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn every_regular_file_is_advised() {
        let dir = scratch_dir("files");
        fs::write(dir.join("meta.json"), b"{}").unwrap();
        fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        fs::create_dir_all(dir.join("nested")).unwrap();
        assert_eq!(purge_dir(&dir).unwrap(), 2);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_empty_directory_is_not_cold() {
        let dir = scratch_dir("empty");
        let error = purge_dir(&dir).expect_err("nothing to evict must be an error");
        assert!(error.contains("no files"), "{error}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_directory_names_the_path() {
        let error = purge_dir(Path::new("/nonexistent/inmotion/artifact"))
            .expect_err("a missing artifact directory must fail");
        assert!(error.contains("/nonexistent/inmotion/artifact"), "{error}");
    }
}
