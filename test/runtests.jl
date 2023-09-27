"""
The majority of tests written for this library are for visual inspection to compare 
results to results taken from literature. 

When quantative measures are available, the comparison functions written will have two 
outputs. The first is data to be used in comparison for @testsets, the other is a plot to 
visualize the comparing data. 
"""

using Puffer
using Test

VISUALIZE = false
include("aoa_tests.jl")
include("garrickComp.jl")
include("teng87_comparison.jl")