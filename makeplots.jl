using JLD2
using SharedArrays, LinearAlgebra, Distributed
using PyPlot
@load "parameters.jld"
@load "data.jld"

figure()
plot(t, omegaEst, label="Bayes")
plot(t, omegaEst + sigmaBayes, "C0--")
plot(t, omegaEst - sigmaBayes, "C0--")
plot(t, omegaMaxLik, label="Max-Lik")
plot(t, fill(omegaTrue, size(t)), label="True")
legend()
savefig("bayes.pdf")
close()

figure()
imshow(probBayes[end:-1:1,:], aspect="auto", extent=[0, Tfinal, omegaMin, omegaMax])
plot(t, fill(omegaTrue,size(t)),"w--")
xlabel("t")
ylabel("ω");
savefig("bayes2.pdf")
close()