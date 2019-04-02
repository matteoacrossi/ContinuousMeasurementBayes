using JSON
symb_dict(d::Dict) = Dict(Symbol(k) => v for (k, v) in d)
expparams = symb_dict.(JSON.parse(open("params.json")))

estparams = Dict(
:omegaMin  => 2., # minimum value of omega
:omegaMax  => 4., # maximum value of omega
:Nomega => 200)