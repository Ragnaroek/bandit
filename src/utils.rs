use std::fs::OpenOptions;
use std::io;
use std::io::{Error, ErrorKind, Write};
use std::path::PathBuf;
use std::time;
use Identifiable;

pub(crate) fn select_argmax(collection: &[f64]) -> Option<usize> {
    let mut current_max_value = None;
    let mut current_max_position = None;
    for (i, x) in collection.iter().enumerate() {
        if current_max_value.unwrap_or(f64::MIN) < *x {
            current_max_value = Some(*x);
            current_max_position = Some(i);
        }
    }
    current_max_position
}

pub(crate) fn log_command<A: Identifiable>(cmd: &str, arm: &A) -> String {
    format!("{};{};{}", cmd, arm.ident(), timestamp())
}

pub(crate) fn timestamp() -> u64 {
    let timestamp_result = time::SystemTime::now().duration_since(time::UNIX_EPOCH);
    let timestamp = timestamp_result.expect("system time");
    timestamp.as_secs() * 1_000 + u64::from(timestamp.subsec_millis())
}

pub(crate) fn log(line: &str, path: &Option<PathBuf>) {
    if path.is_none() {
        return;
    }

    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path.as_ref().unwrap());
    if file.is_ok() {
        let write_result = writeln!(file.unwrap(), "{line}");
        if write_result.is_err() {
            println!("writing log failed {line}");
        }
    } else {
        println!("logging failed: {line}");
    }
}

pub(crate) fn find_arm<'a, A: Identifiable>(arms: &'a [A], ident: &str) -> io::Result<&'a A> {
    for arm in arms {
        if arm.ident() == ident {
            return Ok(arm);
        }
    }
    Err(Error::new(
        ErrorKind::NotFound,
        format!("arm {ident} not found"),
    ))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn select_first_as_max_works() {
        let values = [10., 4., 3., 2.];
        assert_eq!(select_argmax(&values), Some(0))
    }

    #[test]
    fn select_last_as_max_works() {
        let values = [4., 3., 2., 10.];
        assert_eq!(select_argmax(&values), Some(3))
    }

    #[test]
    fn select_works() {
        let values = [0.56, 0.73, 1.67, 0.57];
        assert_eq!(select_argmax(&values), Some(2))
    }
}
