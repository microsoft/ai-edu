import Flux
import Flux: @treelike, param
import Flux.Tracker
import Flux.Tracker: @grad

glorot_normal(dims...) = randn(dims...) .* sqrt(2.0/sum(dims))

struct Dense_m{F,A,V}
    σ::F
    weight::A
    bias::V
    regularizationw::Bool # indicate whether to regularize the weight
    regularizationb::Bool # indicate whether to regularize the bias
end

# two initialize methods
function Dense_m(in::Integer, out::Integer, σ = identity; initw = glorot_normal, initb = zeros)
    return Dense_m(σ, param(initw(out, in)), param(initb(out)), false, false)
end

function Dense_m(in::Integer, out::Integer, regularizationw::Bool, regularizationb::Bool, σ = identity; initw = glorot_normal, initb = zeros)
    return Dense_m(σ, param(initw(out, in)), param(initb(out)), regularizationw, regularizationb)
end

@treelike Dense_m


# since the '*' and '+' has been overloaded in the Flux.Tracker.scalar, so there is no need to rewrite the back function
function (d::Dense_m)(x)
    σ, w, b = d.σ, d.weight, d.bias
    σ.(w * x .+ b)
end
