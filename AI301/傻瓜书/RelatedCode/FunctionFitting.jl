

using Plots

function SampleFun(X)
    Y = 0.4 * x^2 + 0.3 * x *sin(15. * x) + 0.01 * cos(50 .* x) - 0.3
    return Y
end

function Loss(Z,Y,m)
    q = (Z-Y).^2
    #l = sum(q)/2/m
    l = sum(q)/2
    return l
end

function dJw(Z,Y,X,m)
    q = (Z-Y).*X
    #w = sum(q)/m
    w = sum(q)
    return w
end

function dJb(Z,Y,m)
    q = Z-Y
    #b = sum(q)/m
    b = sum(q)
    return b
end

n = 100
m = Int64(n)
X = rand(n)
Noise = rand(n)/5
W = 2
B = 3
Y = W .* X + Noise .+ B

plot(X,Y,seriestype=:scatter)

eta = 0.01
w = Float64(0)
b = Float64(0)
err = 1e-6
lastQ = 10
maxIte = 1000
ite = 0

while ite < maxIte
    global w,b,lastQ, ite
    Z = w .* X .+ b
    Q = Loss(Z,Y,m)

    if abs(Q - lastQ) < err
        break
    end

    dW = dJw(Z,Y,X,m)
    dB = dJb(Z,Y,m)

    w = w - eta * dW
    b = b - eta * dB

    print("ite:$ite\tw:$w\tb:$b\n")

    lastQ = Q
    ite += 1
end

A = w .* X .+ b
plot!(X,A)
