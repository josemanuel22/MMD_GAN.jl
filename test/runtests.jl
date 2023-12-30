using MMD_GAN
using Flux
using Test
using HypothesisTests
using Distributions

tol = 1e-5

include("../src/mmd.jl")


function test_generation_quality(gen, target_model, noise_model, num_samples=1000)
    generated_samples = [gen([rand(noise_model)])[1] for _ in 1:num_samples]
    real_samples = rand(target_model, num_samples)

    # Perform a two-sample Kolmogorov-Smirnov test
    ks_test = ApproximateTwoSampleKSTest(generated_samples, real_samples)
    p_value = pvalue(ks_test)

    return p_value
end
@testset "MMD_GAN.jl" begin
    enc = Chain(Dense(1, 11), elu, Dense(11, 29), elu)
    dec = Chain(Dense(29, 11), elu, Dense(11, 1))
    gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))

    noise_model = Normal(0.0f0, 1.0f0)
    target_model = Normal(4.0f0, 2.0f0)

    hparams = HyperParamsMMD(;
        noise_model=noise_model,
        target_model=target_model,
        data_size=100,
        batch_size=1,
        num_gen=1,
        num_enc_dec=3,
        epochs=1000000,
        lr_dec=1.0e-5,
        lr_enc=1.0e-5,
        lr_gen=1.0e-5,
    )
    gen_losses, dec_losses = train_mmd_gan(enc, dec, gen, hparams)
    @test length(gen_losses) == hparams.epochs * hparams.num_gen
    @test length(dec_losses) == hparams.epochs * hparams.num_enc_dec

    #p_value = test_generation_quality(gen, target_model, noise_model)
    #@test p_value > 0.05
end

@testset "mix_rbf_mmd2" begin
    X = [0.4126 0.334]
    Y = [-0.3432 0.45]
    sigma_list = [1.0, 2.0, 4.0, 8.0, 16.0] ./ 1.0
    @test mix_rbf_mmd2(X, Y, sigma_list) == 0.6955453052850462

    X = [-0.9964 1.6757 -1.0000 -0.9621 3.0564; -0.9966 1.6849 -1.0000 -0.9630 3.0814]
    Y = [-0.4433 0.1915 0.2270 0.0989 -0.1531; -0.4603 0.2200 0.1087 0.0844 -0.1827]
    @test mix_rbf_mmd2(X, Y, sigma_list) == 4.754166275960963

    X = [0.0588 24.6208 0.6140 1.9435 21.2479; 0.0780 23.8120 0.5976 1.8904 20.5289]
    Y = [0.2057 18.4525 0.4892 1.5384 15.7635; 0.1529 20.6706 0.5340 1.6841 17.7357]
    @test mix_rbf_mmd2(X, Y, sigma_list) == 4.65966235831603
end;

@testset "_mmd2" begin
    Kₓₓ = [5.0 4.999526869857079; 4.999526869857079 5.0]
    Kₓᵧ = [2.6194483555739208 2.629665928762427; 2.610644548009624 2.6206924815501655]
    Kᵧᵧ = [5.0 4.989256995960984; 4.989256995960984 5.0]
    @test _mmd2(Kₓₓ, Kₓᵧ, Kᵧᵧ) == 4.748558208869993
    @test _mmd2(Kₓₓ, Kₓᵧ, Kᵧᵧ, false, true) == 4.754166275960963
end;
