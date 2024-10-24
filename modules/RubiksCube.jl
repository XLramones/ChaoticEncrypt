## RubiksCube.jl helper module
# functions to create and manipulate a KxK rubiks cube. Each 'squeare' is an 8bit / UInt8 

module RubiksCube

export initialize_cube, rotate_row!, rotate_col!, display_cube


# Function to initialize the cube
function initialize_cube(n::Int, face_matrices::Dict{String,Matrix{UInt8}}=Dict())
    faces = ["U", "D", "L", "R", "F", "B"]
    default_colors = Dict(
        "U" => 'W',  # White
        "D" => 'Y',  # Yellow
        "L" => 'O',  # Orange
        "R" => 'R',  # Red
        "F" => 'G',  # Green
        "B" => 'B'   # Blue
    )
    cube = Dict{String,Matrix{UInt8}}()
    for face in faces
        if haskey(face_matrices, face)
            cube[face] = face_matrices[face]
            @assert size(cube[face]) == (n, n) "Face $face must be of size $n x $n"
        else
            cube[face] = fill(default_colors[face], n, n)
        end
    end
    return cube
end

# Function to rotate a row
function rotate_row!(cube::Dict{String,Matrix{UInt8}}, row_index::Int, direction::Int, amount::Int)
    n = size(cube["F"], 1)
    @assert 1 ≤ row_index ≤ n "Row index must be between 1 and $n"
    if direction == 1
        for i in 1:amount
            temp = copy(cube["F"][row_index, :])
            cube["F"][row_index, :] = cube["L"][row_index, :]
            cube["L"][row_index, :] = cube["B"][row_index, :]
            cube["B"][row_index, :] = cube["R"][row_index, :]
            cube["R"][row_index, :] = temp
            if row_index == 1
                cube["U"] = rot90(cube["U"], -1)
            elseif row_index == n
                cube["D"] = rot90(cube["D"], 1)
            end
        end
    elseif direction == 0
        for i in 1:amount
            temp = copy(cube["F"][row_index, :])
            cube["F"][row_index, :] = cube["R"][row_index, :]
            cube["R"][row_index, :] = cube["B"][row_index, :]
            cube["B"][row_index, :] = cube["L"][row_index, :]
            cube["L"][row_index, :] = temp
            if row_index == 1
                cube["U"] = rot90(cube["U"], 1)
            elseif row_index == n
                cube["D"] = rot90(cube["D"], -1)
            end
        end
    else
        error("Invalid direction: use \"0:left\" or \"1:right\"")
    end
end

# Function to rotate a column
function rotate_col!(cube::Dict{String,Matrix{UInt8}}, col_index::Int, direction::Int, amount::Int)
    n = size(cube["F"], 2)
    @assert 1 ≤ col_index ≤ n "Column index must be between 1 and $n"
    opposite_col_index = n - col_index + 1
    if direction == 0
        for i in 1:amount
            temp = copy(cube["F"][:, col_index])
            cube["F"][:, col_index] = cube["D"][:, col_index]
            cube["D"][:, col_index] = reverse(cube["B"][:, opposite_col_index])
            cube["B"][:, opposite_col_index] = reverse(cube["U"][:, col_index])
            cube["U"][:, col_index] = temp
            if col_index == 1
                cube["L"] = rot90(cube["L"], 1)
            elseif col_index == n
                cube["R"] = rot90(cube["R"], -1)
            end
        end
    elseif direction == 1
        for i in 1:amount
            temp = copy(cube["F"][:, col_index])
            cube["F"][:, col_index] = cube["U"][:, col_index]
            cube["U"][:, col_index] = reverse(cube["B"][:, opposite_col_index])
            cube["B"][:, opposite_col_index] = reverse(cube["D"][:, col_index])
            cube["D"][:, col_index] = temp
            if col_index == 1
                cube["L"] = rot90(cube["L"], -1)
            elseif col_index == n
                cube["R"] = rot90(cube["R"], 1)
            end
        end
    else
        error("Invalid direction: use \"0:up\" or \"1:down\"")
    end
end

# Function to display the cube faces
function display_cube(cube::Dict{String,Matrix{UInt8}})
    n = size(cube["U"], 1)  # Assuming all faces are n x n

    # Function to convert a row of a face to a string with spaces
    function row_to_str(row::Vector{UInt8})
        return join(row, " ")
    end

    # Prepare each face's rows as strings
    U_rows = [row_to_str(cube["U"][i, :]) for i in 1:n]
    L_rows = [row_to_str(cube["L"][i, :]) for i in 1:n]
    F_rows = [row_to_str(cube["F"][i, :]) for i in 1:n]
    R_rows = [row_to_str(cube["R"][i, :]) for i in 1:n]
    B_rows = [row_to_str(cube["B"][i, :]) for i in 1:n]
    D_rows = [row_to_str(cube["D"][i, :]) for i in 1:n]

    # Determine the width for padding
    face_width = 2 * n - 1  # Each cell separated by a space

    # Create padding for the U and D faces
    padding = " "^(face_width + 2)

    # Print U face
    println(padding, "U face:")
    for row in U_rows
        println(" "^((face_width + 2) ÷ 2), row)
    end
    println()

    # Print L, F, R, B faces side by side
    println(
        rpad("L face:", face_width + 2),
        rpad("F face:", face_width + 2),
        rpad("R face:", face_width + 2),
        rpad("B face:", face_width + 2)
    )
    for i in 1:n
        println(
            rpad(L_rows[i], face_width + 2),
            rpad(F_rows[i], face_width + 2),
            rpad(R_rows[i], face_width + 2),
            rpad(B_rows[i], face_width + 2)
        )
    end
    println()

    # Print D face
    println(padding, "D face:")
    for row in D_rows
        println(" "^((face_width + 2) ÷ 2), row)
    end
    println()
end
function rot90(matrix::Matrix, k::Int=1)::Matrix
    k = mod(k,4)
    if k == 0
        return matrix
    elseif k == 1
        # 90 degrees Counterclockwise
        return reverse(transpose(matrix), dims=2)
    elseif k == 2
        # 180 degrees
        return reverse(reverse(matrix, dims=1), dims=2)
    elseif k == 3
        # 270 degrees Counterclockwise (or 90 degrees Clockwise)
        return reverse(transpose(matrix), dims=1)
    end
end
end  # End of module


