using BenchmarkTools
using Dynare
using FastLapackInterface
using LinearAlgebra
using PolynomialMatrixEquations
using Test

include("../src/rendahl.jl")
context = @dynare "irbc.mod"

model = context.models[1]
results = context.results.model_results[1]
work = context.work
endogenous = results.trends.endogenous_steady_state
exogenous = results.trends.exogenous_steady_state
steadystate = results.trends.endogenous_steady_state
params = work.params
df = context.dynarefunctions
ncol = model.n_bkwrd + model.n_current + model.n_fwrd + 2 * model.n_both
tmp_nbr = df.dynamic!.tmp_nbr::Vector{Int64}
ws = Dynare.DynamicWs(model.endogenous_nbr, model.exogenous_nbr, ncol, sum(tmp_nbr[1:2]))

# abbreviations
LRE = Dynare.LinearRationalExpectations
LREWs = Dynare.LinearRationalExpectationsWs

LRE_results = results.linearrationalexpectations

jacobian =
    Dynare.get_dynamic_jacobian!(ws, params, endogenous, exogenous, steadystate, model, context.dynarefunctions, 2)
wsLRE = LREWs(
    "CR",
    model.endogenous_nbr,
    model.exogenous_nbr,
    model.exogenous_deterministic_nbr,
    model.i_fwrd_b,
    model.i_current,
    model.i_bkwrd_b,
    model.i_both,
    model.i_static,
)
LRE.remove_static!(jacobian, wsLRE)
LRE.get_abc!(wsLRE, jacobian)
A0 = wsLRE.a
B0 = wsLRE.b
C0 = wsLRE.c

n = size(wsLRE.a, 1)

A = copy(A0)
B = copy(B0)
C = copy(C0)

wsr = RendahlWs(n)
X1 = zeros(n, n)
ENV["JULIA_DEBUG"] = Main
rendahl_solve!(X1, C, B, A, wsr::RendahlWs, maxiter=1000, tol=1e-6)
ENV["JULIA_DEBUG"] = ""
display(X1)
@show norm(A0*X1*X1 + B0*X1 + C0)

@btime rendahl_solve!(X1, C, B, A, wsr::RendahlWs, maxiter=1000, tol=1e-6)

A = copy(A0)
B = copy(B0)
C = copy(C0)
X2 = zeros(n, n)
wsc = CyclicReductionWs(n)
cyclic_reduction!(X2, C, B, A, wsc, 1e-6, 1000)
display(X2)
@show norm(A0*X2*X2 + B0*X2 + C0)

@btime cyclic_reduction!(X2, C, B, A, wsc, 1e-6, 1000)


