macro_rules! check_hypre {
    ( $res:expr) => {{
        match $res {
            0 => {}
            res => {
                return Err(HypreError::new(res));
            }
        }
    }};
}
