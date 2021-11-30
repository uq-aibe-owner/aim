#u + β * w

using Ipopt, ForwardDiff

using ReverseDiff:
    GradientTape,
    GradientConfig,
    gradient,
    gradient!,
    JacobianTape,
    JacobianConfig,
    jacobian,
    jacobian!,
    hessian,
    hessian!,
    HessianTape,
    HessianConfig,
    compile,
    DiffResults

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)


#=const D = 2
const czero = ones(D)
const szero = ones(D,D)
# the variables
u = Float64
w = Float64
c = similar(czero)
l = similar(czero)
q = similar(czero)
kp = similar(czero)
sComb = similar(czero)
mComb = similar(czero)
s = similar(szero)
m = similar(szero)
x = Float64[]
push(x, u)
=#

function F(x)
    u + β * w
end

 xzero = [1., 2., 3., 4.]
 FgradTape = GradientTape(F, xzero)
 compFgradTape = compile(FgradTape)
function gradF(x)
    return gradient!(similar(xzero), compFgradTape, x)
end

function G(x)
    [x[1] * x[2] * x[3] * x[4],
     x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2]
end

 GgradTape = [GradientTape(x -> G(x)[1], xzero), GradientTape(x -> G(x)[2], xzero)]
 compGgradTape = [compile(GgradTape[1]), compile(GgradTape[2])]
function jacG(x)
    results = similar(xzero, m, n)
    results[1,:] = gradient!(similar(xzero), compGgradTape[1], x)
    results[2,:] = gradient!(similar(xzero), compGgradTape[2], x)
    return results
end

FhessTape = HessianTape(F, xzero)
compFhessTape = compile(FhessTape)
function hessF(x)
    return hessian!(similar(xzero, n, n), compFhessTape, x)
end

GhessTape = [HessianTape(x -> G(x)[1], xzero), HessianTape(x -> G(x)[2], xzero)]
compGhessTape = [compile(GhessTape[1]), compile(GhessTape[2])]
function vHessG(x)
    return [hessian!(similar(xzero, n, n), compGhessTape[1], x),
            hessian!(similar(xzero, n, n), compGhessTape[2], x)]
end
#we now define the callback function for ipopt
function eval_f(x)
    F(x)
end

function eval_g(x, g)
  # Bad: g    = zeros(2)  # Allocates new array
  # OK:  g[:] = zeros(2)  # Modifies 'in place'
    g[1] = G(x)[1]
    g[2] = G(x)[2]
end

function eval_grad_f(x, grad_f)
  # Bad: grad_f    = zeros(4)  # Allocates new array
  # OK:  grad_f[:] = zeros(4)  # Modifies 'in place'
  grad_f[1] = gradF(x)[1]
  grad_f[2] = gradF(x)[2]
  grad_f[3] = gradF(x)[3]
  grad_f[4] = gradF(x)[4]
end

function eval_jac_g(x, mode, rows, cols, values)
  if mode == :Structure
    # Constraint (row) 1
    rows[1] = 1; cols[1] = 1
    rows[2] = 1; cols[2] = 2
    rows[3] = 1; cols[3] = 3
    rows[4] = 1; cols[4] = 4
    # Constraint (row) 2
    rows[5] = 2; cols[5] = 1
    rows[6] = 2; cols[6] = 2
    rows[7] = 2; cols[7] = 3
    rows[8] = 2; cols[8] = 4
  else
    # Constraint (row) 1
#    values[1:n] = jacG(x)[1,:]
#    values[1] = jacG(x)[1, 1]
#    values[2] = jacG(x)[1, 2]
#    values[3] = jacG(x)[1, 3]
#    values[4] = jacG(x)[1, 4]
    # Constraint (row) 2
#     values[n+1:2n] = jacG(x)[2,:]
#    values[5] = 2*x[1]  # 2,1
#    values[6] = 2*x[2]  # 2,2
#    values[7] = 2*x[3]  # 2,3
#    values[8] = 2*x[4]  # 2,4
      values[:] = vec(transpose(jacG(x)))
  end
end

function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
  if mode == :Structure
    # Symmetric matrix, fill the lower left triangle only
    idx = 1
    for row = 1:4
      for col = 1:row
        rows[idx] = row
        cols[idx] = col
        idx += 1
      end
    end
  else
    # Again, only lower left triangle
    # Objective
    iter1 = 1
    for i in 1:n
        for j in 1:i
            values[iter1] = obj_factor * hessF(x)[i, j]
            iter1 += 1
        end
    end


#    values[:] = vec(hessF(x)[1:])
#    values[1] = obj_factor * (2*x[4])  # 1,1
#    values[2] = obj_factor * (  x[4])  # 2,1
#    values[3] = 0                      # 2,2
#    values[4] = obj_factor * (  x[4])  # 3,1
#    values[5] = 0                      # 3,2
#    values[6] = 0                      # 3,3
#    values[7] = obj_factor * (2*x[1] + x[2] + x[3])  # 4,1
#    values[8] = obj_factor * (  x[1])  # 4,2
#    values[9] = obj_factor * (  x[1])  # 4,3
#    values[10] = 0                     # 4,4
#
    # Both constraints
    iter1 = 1
    for i in 1:n
        for j in 1:i
            values[iter1] += lambda[1] * vHessG(x)[1][i, j] + lambda[2] * vHessG(x)[2][i, j]
            iter1 += 1
        end
    end
    # First constraint
#    values[2] += lambda[1] * (x[3] * x[4])  # 2,1
#    values[4] += lambda[1] * (x[2] * x[4])  # 3,1
#    values[5] += lambda[1] * (x[1] * x[4])  # 3,2
#    values[7] += lambda[1] * (x[2] * x[3])  # 4,1
#    values[8] += lambda[1] * (x[1] * x[3])  # 4,2
#    values[9] += lambda[1] * (x[1] * x[2])  # 4,3
#    for i in 1:n
#        for j in 1:i
#            values[iter1] += lambda[2] * vHessG(x)[i + n, j]
#            iter1 += 1
#        end
#    end

    # Second constraint
#    values[1]  += lambda[2] * 2  # 1,1
#    values[3]  += lambda[2] * 2  # 2,2
#    values[6]  += lambda[2] * 2  # 3,3
#    values[10] += lambda[2] * 2  # 4,4
  end
end

n = 4
x_L = [1.0, 1.0, 1.0, 1.0]
x_U = [5.0, 5.0, 5.0, 5.0]

m = 2
g_L = [25.0, 40.0]
g_U = [2.0e19, 40.0]

function probcall()
prob = createProblem(n, x_L, x_U, m, g_L, g_U, 8, 10,
                     eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h);
prob.x = [1.0, 5.0, 5.0, 1.0];
status = solveProblem(prob);
println(Ipopt.ApplicationReturnStatus[status])
println(prob.x)
println(prob.obj_val)
end
#if @isdefined(obj)
#else
#    obj = Float64[]
#end
#if length(obj) == 1
#else
#   println(obj[length(obj)-1])
#end
