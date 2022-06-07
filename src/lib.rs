use enum_dispatch::enum_dispatch;

mod matrix;
mod solver;

#[enum_dispatch]
trait LinearSolver {
    fn solve(self);
    fn set_precond(&mut self);
}

struct CGSolver;

impl LinearSolver for CGSolver {
    fn solve(self) {
        todo!()
    }

    fn set_precond(&mut self) {
        todo!()
    }
}

struct AMGSolver;

impl LinearSolver for AMGSolver {
    fn solve(self) {
        todo!()
    }

    fn set_precond(&mut self) {
        todo!()
    }
}

#[enum_dispatch(LinearSolver)]
enum Solver {
    CG(CGSolver),
    AMG(AMGSolver),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let solver: Solver = CGSolver {}.into();
        solver.solve();
    }
}
