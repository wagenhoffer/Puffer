using Coverage
# process '*.cov' files
coverage = process_folder() # defaults to src/; alternatively, supply the folder name as argument
# coverage = append!(coverage, process_folder("deps"))  # useful if you want to analyze more than just src/
# process '*.info' files, if you collected them
coverage = merge_coverage_counts(coverage, filter!(
    let prefixes = (joinpath(pwd(), "src", ""),
                    joinpath(pwd(), "deps", ""))
        c -> any(p -> startswith(c.filename, p), prefixes)
    end,
    LCOV.readfolder("test")))
# Get total coverage for all Julia files
covered_lines, total_lines = get_summary(coverage)
# Or process a single file
@show get_summary(process_file(joinpath("src", "BemRom.jl")))
println("($(covered_lines/total_lines * 100)%) covered") 

# cd(joinpath(@__DIR__, "..", "..")) do
#     covered_lines, total_lines = get_summary(process_folder())
#     percentage = covered_lines / total_lines * 100
#     println("($(percentage)%) covered")
# end
# cd(joinpath(@__DIR__, "..", "..")) do
#     Codecov.submit(Codecov.process_folder())
# end