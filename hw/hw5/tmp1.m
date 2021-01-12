tens_struct_data;

c = g * m
A1 = A(1:p,:)
A2 = A(p+1:n,:)
cbar = c(p+1:n)
cvx_begin
	variable k(N)
	D = diag(k);
	bx = A1'*x_fixed;
	by = A1'*y_fixed;
	vx = A2*D*bx;
	vy = A2*D*by + cbar;
	maximize(bx'*D*bz - matrix_frac(vx, A2*D*A2') + ...
			by'*D'by - matrix(vy,A2*D*A2'))
	subject to
		k >= 0; sum(k) == k_tot;
cvx_end

Eunif = 0.5 * x_unif' * A * diag(k_unif) * A' * x_unif;
Eunif = Eunif + 0.5 * y_unif' * A * diag(k_unif) * A' * y_unif';
Eunif = Eunif + c' * y_unif
Emin = 0.5 * cvx_optval + c(1:p)' * y_fixed

xmin = -(A2 * D * A2') \ (A2 * D * A1' * x_fixed);
ymin = -(A2 * D * A2') \ (A2 * D * A1' * y_fixed + cbar);

xopt = zeros(n, 1);
xopt(1:p) = x_fixed;
xopt(p + 1:n) = xmin;

yopt = zeros(n, 1);
yopt(1:p) = y_fixed;
yopt(p + 1: n) = ymin;
