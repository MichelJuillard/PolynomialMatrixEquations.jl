using Dynare
include("../../FastLapackInterface.jl/src/FastLapackInterface.jl")
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
ws = DynamicWs(model.endogenous_nbr, model.exogenous_nbr, ncol, sum(tmp_nbr[1:2]))

# abbreviations
LRE = Dynare.LinearRationalExpectations
LREWs = Dynare.LinearRationalExpectationsWs

LRE_results = results.linearrationalexpectations

jacobian =
    get_dynamic_jacobian!(ws, params, endogenous, exogenous, steadystate, model, context.dynarefunctions, 2)
algo = options.dr_algo
wsLRE = LREWs(
    "RE",
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

n = size(LRE.a)
wsr = RendahlWs(n)
rendahl_solve!(X, LRE.a, LRE.b, LRE.c, ws::RendahlWs, maxiter=1000, tol=1e-6)
    
