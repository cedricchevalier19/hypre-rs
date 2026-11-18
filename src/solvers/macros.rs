//! Helper macros for setting, reading and validating solver parameters.

/// Calls the hypre setter `$func` on `$obj` if `$param` is `Some`, converting
/// the value to the setter's type; returns early on a conversion or hypre
/// error.
macro_rules! set_parameter {
    ( $func:expr, $obj:expr, $param:expr ) => {{
        if let Some(p_value) = $param {
            check_hypre!(unsafe { $func($obj, p_value.try_into()?) });
        }
    }};
}

/// Calls the hypre getter `$func` on `$obj`, reading a `$t`, and converts it
/// to the expected type. Evaluates to a `HypreResult`.
macro_rules! get_parameter {
    ( $func:expr, $obj:expr, $t:ty ) => {{
        let mut p_t: $t = Default::default();
        let err = unsafe { $func($obj, &mut p_t) };
        if err != 0 {
            Err(HypreError::new(err))
        } else {
            p_t.try_into().map_err(HypreError::from)
        }
    }};
}

/// Returns [HypreError::InvalidParameterPositive] from a builder's
/// `validate` if the optional parameter `$param` is set and negative.
macro_rules! check_positive_parameter {
    ( $obj:expr, $param:ident) => {{
        if let Some(Some($param)) = $obj.$param {
            if $param < 0.into() {
                return Err(HypreError::InvalidParameterPositive);
            }
        }
    }};
}
