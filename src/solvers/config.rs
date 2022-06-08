#[derive(Default, Debug)]
pub struct GenericIterativeSolverConfig {
    pub tol: Option<f64>,
    pub abs_tol: Option<f64>,
    pub res_tol: Option<f64>,
    pub abs_tol_fact: Option<f64>,
    pub conv_tol_fact: Option<f64>,
    pub stop_crit: Option<usize>,
    pub max_iters: Option<usize>,
    pub two_norm: Option<bool>,
    pub rel_change: Option<bool>,
    pub logging: Option<u32>,
    pub print_level: Option<u32>,
}

impl GenericIterativeSolverConfig {
    pub fn validate(&self) -> bool {
        let mut check = self.tol.map_or_else(|| false, |x| x > 0.0);
        check &= self.abs_tol.map_or_else(|| false, |x| x > 0.0);
        check
    }
}
