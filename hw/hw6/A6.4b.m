team_data
A1 = sparse(1:m, train(:, 1), train(:, 3), m, n);
A2 = sparse(1:m, train(:, 2), -train(:, 3), m, n);
A = A1 + A2;

cvx_begin
	variable a_hat(n)
	minimize(-sum(log_normcdf(A * a_hat/sigma)))
	subject to
		a_hat >= 0
		a_hat <= 1
cvx_end

a_hat
