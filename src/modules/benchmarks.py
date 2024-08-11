import cvxpy as cp

class MultiPeriodMeanVariance:
    """A Portfolio Optimization Algorithm using Harry Markowitz's Mean Variance Framework.

    The Algorithm uses CVXPY to perform a Multi-Period Convex Optimization to determine the near-optimal set
    of weights by:
        maximizing the expected returns - portfolio volatility - trade cost - holding cost
    over a multi-time step horizon.
    """
    def __init__(self, 
                 num_assets:int,
                 multi_period_step:int,
                 min_individual_allocation:int, 
                 max_individual_allocation:int, 
                 risk_aversion_coefficient:int=0.5, 
                 trade_cost:int=0.0025, 
                 holding_cost:int=0):

        """Initializes the Multi-Period Markowitz's Optimization instance.

        Args:
            num_assets: Number of assets in the portfolio.
            multi_period_step: Number of time-steps over to optimize.
            min_individual_allocation: Minimum percentage allocation over a single stock. 
            max_individual_allocation: Maximum percentage allocation to prevent over allocation into a single stock.
            risk_aversion_coefficient: Investors risk tolerance coefficient.
        """

        self._num_assets = num_assets
        self._multi_period_step = multi_period_step
        self._trade_cost = trade_cost
        self._holding_cost = holding_cost
        self._min_allocation = min_individual_allocation
        self._max_allocation = max_individual_allocation
        self._risk_aversion_coefficient = risk_aversion_coefficient
        self._total_weights = 1 # prevents shorting and leveraging

    def __call__(self, mu_returns, cov_matrices):
        T = mu_returns.shape[0]
        num_assets = mu_returns.shape[1]
        assert(T == self._multi_period_step)
        assert(num_assets == self._num_assets)

        multi_period_optimal_weights = self.mv_optimization(T, num_assets, mu_returns, cov_matrices)
        return multi_period_optimal_weights

    def mv_optimization(self, T, num_assets, mu_returns, cov_matrices):
        weights = cp.Variable((T, num_assets))
        constraints = []
        for i in range(T):
            constraints.append(weights[i,:] <= self._max_allocation) 
            constraints.append(weights[i,:] >= self._min_allocation)
            constraints.append(cp.sum(weights[i,:]) == self._total_weights)

        objective = 0
        for i in range(T):
            portfolio_return = (weights[i]@mu_returns[i]).sum()
            portfolio_risk = self._risk_aversion_coefficient*cp.quad_form(weights[i], cov_matrices[i])
            portfolio_objective_i = portfolio_return - portfolio_risk
            objective += portfolio_objective_i

        portfolio_rebalancing_cost = self._trade_cost*cp.abs((weights[1:,:] - weights[:-1,:])).sum()
        objective -= portfolio_rebalancing_cost

        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve()

        optimal_weights = weights.value
        return optimal_weights
