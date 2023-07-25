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

macro_rules! check_positive_parameter {
    ( $obj:expr, $param:ident) => {{
        if let Some(Some($param)) = $obj.$param {
            if $param < 0.into() {
                return Err(HypreError::InvalidParameterPositive);
            }
        }
    }};
}
