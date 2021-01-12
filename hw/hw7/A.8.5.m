n = 4000;
k = 100;
delta = 1;
eta = 1;

A = rand(k, n);
b = rand(k, 1);

e = ones(n, 1);
D = spdiags([-e 2*e -e],[-1 0 1], n,n);
D(1, 1) = 1;
D(n, n) = 1;
I = sparse(1:n ,1:n ,1);
F = A' * A + eta * I + delta * D;
P = eta * I + delta * D;
g = A' * b;

fprintf('\nComputing solution directly\n');
s1 = cputime;
x_gen = F \ g;
s2 = cputime;
fprintf('Done (in %g sec)\n',s2-s1);


fprintf('\nComputing solution using efficient method\n');

%x_eff = P^{-1}g - P^{-1}A’(I +AP^{-1}A’)^{-1}AP^{-1}g.
t1= cputime;
Z_0 = P \ [g A'];
z_1 = Z_0(:, 1);
Z_2 = Z_0(:, 2:k+1);
z_3 = (sparse(1:k, 1:k, 1) + A * Z_2) \ (A * z_1);
x_eff = z_1 - Z_2 * z_3;
t2 = cputime;
fprintf('Done (in %g sec)\n',t2-t1);
fprintf('\nrelative error = %e\n',norm(x_eff-x_gen)/norm(x_gen));
