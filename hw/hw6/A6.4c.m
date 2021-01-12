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

A1_test = sparse(1:m_test, test(:,1), 1, m_test, n);
A2_test = sparse(1:m_test, test(:,2), -1, m_test, n);
A_test = A1_test + A2_test;

result = sign(A_test * a_hat)
Pml = 1 - length(find(res - test(:, 3))) / m_test;
Ply = 1 - length(find(train(:, 3) - test(:, 3))) / m_test;
