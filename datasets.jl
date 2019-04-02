using JSON
symb_dict(d::Dict) = Dict(Symbol(k) => v for (k, v) in d)
experimental_params = symb_dict.(JSON.parse(open("params.json")))