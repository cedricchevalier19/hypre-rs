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

macro_rules! set_parameter {
    ( $func:path, $obj:expr, $param:expr ) => {{
        if let Some(p_value) = $param {
            check_hypre!(unsafe { $func($obj, p_value.try_into().unwrap()) });
        }
    }};
}

macro_rules! get_parameter {
    ( $func:path, $obj:expr, $t:ty ) => {{
        let mut p_t: $t = Default::default();
        match unsafe { $func($obj, &mut p_t) } {
            0 => Ok(p_t.try_into().unwrap()),
            err => Err(HypreError::new(err)),
        }
    }};
}
