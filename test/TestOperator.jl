module TestOperator
    using Test
    include("../src/Operator.jl")

    function main()
        @test Operator.soft_thresholding([2.0], 1.5) == [0.5]
        @test Operator.soft_thresholding([-1.0], 0.5) == [-0.5]
        @test Operator.soft_thresholding([2.0], 0.0) == [2.0]
        @test Operator.soft_thresholding([2.0], 3.0) == [0.0]

        x = [1.0, 1.5, -3.0, 0.5]
        y = [0.0, 0.5, -2.0, 0.0]
        @test Operator.soft_thresholding(x, 1.0) == y

        tau = [0.5, 1.0, 2.0, 1.0]
        z = [0.5, 0.5, -1.0, 0.0]
        @test Operator.soft_thresholding(x, tau) == z
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    using. TestOperator
    TestOperator.main()
end