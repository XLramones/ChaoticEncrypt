# Author: Joshua Baillie
# University of Cape Town
# choatic Confusion-Rubik's cube transformation-Diffusion alogorith implemenation based on https://doi.org/10.1016/j.ijleo.2022.170342 (Zhao et al)
using Images, SHA, FileIO, ImageShow, Printf, ImageIO, JSON
using BenchmarkTools
using Base.Threads
push!(LOAD_PATH, "modules")
using RubiksCube


# parellelized 
function main_encrypt(img, n, x0, y0, u, k)

        K = size(img, 1)

        # normalize he image to an array of values from 0 to 255
        norm_img = clamp.(round.(Int, img .* 255), 0, 255)
        image_bytes = Vector{UInt8}(vec(norm_img))
        println("computing sha")
        hash = bytes2hex(sha256(image_bytes))
        println("computing initconds")
        new_x0, new_y0, new_k, new_u = tweak_inits(hash, x0, y0, u, k)
        println("computing 2DLCCM")
        k_pow = 10.0^new_k
        total_iters = n * K^2 + 999
        # allocate chaos matrix for performance boost
        chaos = Matrix{Float64}(undef, total_iters + 1, 2)
        chaos[1, :] = [new_x0, new_y0]

        if new_u > 700 * 4 || new_k > 14.3 #k to big needs high precision
                k_powBig = BigFloat(10.0)^new_k
                @inbounds for i in 1:total_iters
                        LCCM_bigfloat!(chaos, i, new_u, k_powBig)
                end
        else
                # fill chaos matrix
                @inbounds for i in 1:total_iters
                        LCCM_faster!(chaos, i, new_u, k_pow)
                end
        end
        x_coords = chaos[:, 1][1001:end]
        y_coords = chaos[:, 2][1001:end]

        println("computing brownian steps")
        coords = brownian_motion(x_coords, y_coords, n, K)
        println("snapping coords to grid")
        x_new, y_new = snap_coords_parallel(coords, K)

        println("constructing new image")
        newimg = deconstruct_image(K, x_new, y_new, img)

        norm_newimg = UInt8.(clamp.(round.(Int, newimg .* 255), 0, 255))
        println("mapping onto hexahedron, shuffling")
        X2, X22, Y2, Y22 = extract_sequences(chaos, K)

        faces = create_faces(K, X2, X22, Y2, Y22)
        seq = create_seq(K, X2, X22, Y2, Y22)
        cube = create_cube(norm_newimg, faces...)
        shuffle_cube!(cube, seq)
        println("Calculating Diffusion Matrices")
        A1, A2 = chaos_matrices(K, X2, X22, Y2, Y22)
        Q2h = extract_high_nibble.(cube["F"])
        Q2l = extract_low_nibble.(cube["F"])
        Q3ld = Matrix{UInt8}(undef, K, K)
        Q3hd = Matrix{UInt8}(undef, K, K)
        println("Diffusing")
        Q3 = diffuse(k, new_k, K, A1, A2, Q2h, Q2l, Q3hd, Q3ld)
        newimg3 = zeros(Gray, K, K)
        newimg3 .= clamp.((Q3 ./ 255), 0.0, 1.0)
        key = Dict("hash" => hash, "facesd" => [cube["U"], cube["D"], cube["L"], cube["R"], cube["B"]], "k" => k, "u" => u, "x0" => x0, "y0" => y0)
        return newimg3, key

end
# parellelized 
function main_decrypt(img, key, n)
        println("loading key")
        K = size(img, 1)
        k = key["k"]
        u = key["u"]
        x0 = key["x0"]
        y0 = key["y0"]
        hash = key["hash"]
        ## TODO optimise this
        faces = Vector{Any}(undef, 5)
        for i in 1:5
                facei = key["facesd"][i]
                ## ? why doesnt the below work ?
                # faces[i] = vcat(facei...)
                faces[i] = UInt8.(reshape(vcat(facei...), K, K))
        end
        # normalize he image to an array of values from 0 to 255
        norm_img = UInt8.(clamp.(round.(Int, img .* 255), 0, 255))
        println("computing initconds")
        new_x0, new_y0, new_k, new_u = tweak_inits(hash, x0, y0, u, k)
        println("computing 2DLCCM")
        k_pow = 10.0^new_k
        total_iters = n * K^2 + 999
        # allocate chaos matrix for performance boost
        chaos = Matrix{Float64}(undef, total_iters + 1, 2)
        chaos[1, :] = [new_x0, new_y0]

        if new_u > 700 * 4 || new_k > 14.3 #k to big needs high precision
                k_powBig = BigFloat(10.0)^new_k
                @inbounds for i in 1:total_iters
                        LCCM_bigfloat!(chaos, i, new_u, k_powBig)
                end
        else
                # fill chaos matrix
                @inbounds for i in 1:total_iters
                        LCCM_faster!(chaos, i, new_u, k_pow)
                end
        end
        X2, X22, Y2, Y22 = extract_sequences(chaos, K)
        A1, A2 = chaos_matrices(K, X2, X22, Y2, Y22)
        Q3hd = extract_high_nibble.(norm_img)
        Q3ld = extract_low_nibble.(norm_img)
        Q2l = Matrix{UInt8}(undef, K, K)
        Q2h = Matrix{UInt8}(undef, K, K)
        println("reverting Diffusion")
        Q3 = refuse(k, new_k, K, A1, A2, Q2h, Q2l, Q3hd, Q3ld)
        println("mapping onto hexahedron, unshuffling")
        X2, X22, Y2, Y22 = extract_sequences(chaos, K)
        seq = create_seq(K, X2, X22, Y2, Y22)
        cube = create_cube(Q3, faces...)
        unshuffle_cube!(cube, seq)
        println("computing brownian steps")
        x_coords = chaos[:, 1][1001:end]
        y_coords = chaos[:, 2][1001:end]

        coords = brownian_motion(x_coords, y_coords, n, K)
        println("snapping coords to grid")
        x_new, y_new = snap_coords_parallel(coords, K)

        println("reconstructing image")
        Q2 = clamp.((cube["F"] ./ 255), 0.0, 1.0)
        newimg = zeros(eltype(img), K, K)
        newimg .= Q2
        Q1 = construct_image(K, x_new, y_new, newimg)
        return Q1

end


@inline function LCCM_faster!(chaos_matrix::Matrix{Float64}, i::Int, u::Float64, k_pow::Float64)
        # @fastmath begin
        xi = chaos_matrix[i, 1]
        yi = chaos_matrix[i, 2]
        exp_term_x = exp(u * yi * (1.0 - yi))
        acos_xi = acos(xi)
        temp_x = cos(exp_term_x * acos_xi) * k_pow
        chaos_matrix[i+1, 1] = temp_x - floor(temp_x)
        exp_term_y = exp(u * xi * (1.0 - xi))
        acos_yi = acos(yi)
        temp_y = cos(exp_term_y * acos_yi) * k_pow
        chaos_matrix[i+1, 2] = temp_y - floor(temp_y)
end
@inline function LCCM_bigfloat!(chaos_matrix::Matrix{Float64}, i::Int, u::Float64, k_pow::BigFloat)
        # @fastmath begin
        xi = chaos_matrix[i, 1]
        yi = chaos_matrix[i, 2]
        exp_term_x = exp(u * yi * (1.0 - yi))
        acos_xi = acos(xi)
        temp_x = cos(exp_term_x * acos_xi) * k_pow
        chaos_matrix[i+1, 1] = temp_x - floor(temp_x)
        exp_term_y = exp(u * xi * (1.0 - xi))
        acos_yi = acos(yi)
        temp_y = cos(exp_term_y * acos_yi) * k_pow
        chaos_matrix[i+1, 2] = temp_y - floor(temp_y)
end

function extract_sequences(chaos, K)
        X2 = chaos[:, 1][1:K^2]
        Y2 = chaos[:, 2][1:K^2]
        X22 = X2 .^ 2
        Y22 = Y2 .^ 2
        return X2, X22, Y2, Y22
end

function create_faces(K, X2, X22, Y2, Y22)
        S1 = reshape(UInt8.(mod.(Int.(floor.((X22 .- Y22 .+ 1) .* 10^8)), 256)), K, K)
        S2 = reshape(UInt8.(mod.(Int.(floor.((X2 .- Y22 .+ 1) .* 10^8)), 256)), K, K)
        S3 = reshape(UInt8.(mod.(Int.(floor.((X22 .- Y2 .+ 1) .* 10^8)), 256)), K, K)
        S4 = reshape(UInt8.(mod.(Int.(floor.((X2 .- Y2 .+ 1) .* 10^8)), 256)), K, K)
        S5 = reshape(UInt8.(mod.(Int.(floor.((X22 .- Y2 .^ 3 .+ 1) .* 10^8)), 256)), K, K)
        return (S1, S2, S3, S4, S5)
end

function create_seq(K, X2, X22, Y2, Y22)
        S6 = vcat(elem_to_8bit.(X2)...)[1:1000]
        S7 = vcat(elem_to_8bit.(Y2)...)[1:1000]
        S8 = mod.(Int.(floor.(X2[1:1000] .* 10^8)), K) .+ 1
        S9 = mod.(Int.(floor.(Y2[1:1000] .* 10^8)), 4) .+ 1
        return hcat(S6, S7, S8, S9)

end

function create_cube(img, S1, S2, S3, S4, S5)
        K = size(img, 1)
        faces_dict = Dict{String,Matrix{UInt8}}(
                "U" => S1, "D" => S2, "L" => S3, "R" => S4, "F" => img, "B" => S5
        )

        image_rubik = initialize_cube(K, faces_dict)
        return image_rubik
end

function shuffle_cube!(cube, seq)
        for i in 1:size(seq, 1)
                control = seq[i, :]
                if Bool(control[1])
                        rotate_col!(cube, control[3], control[2], control[4])
                else
                        rotate_row!(cube, control[3], control[2], control[4])
                end
        end
end

function unshuffle_cube!(cube, seq)
        for i in size(seq, 1):-1:1
                control = seq[i, :]
                if Bool(control[1])
                        rotate_col!(cube, control[3], 1 - control[2], control[4])
                else
                        rotate_row!(cube, control[3], 1 - control[2], control[4])
                end
        end
end


function elem_to_8bit(x::Float64)::Vector{Int}
        bits = Vector{Int}(undef, 8)
        for i in 1:8
                bits[i] = (round(Int, x * 255) >> (8 - i)) & 1
        end
        return bits
end

function expand_img(img)
        width, height = size(img, 1), size(img, 2)
        dim = max(width, height)
        expImg = zeros(eltype(img), dim, dim)
        # fill the KxK zero array with the origianl image
        expImg[1:width, 1:height] = img
        return expImg
end

function deconstruct_image(K, x::Vector{Int}, y::Vector{Int}, img)
        newimg = zeros(eltype(img), K, K)
        # flatten the image for efficient indexing
        img_flat = vec(img)
        # new linear indices
        new_indices = (y .- 1) * K + x
        newimg .= reshape(img_flat[new_indices], K, K)
        return newimg
end

function construct_image(K, x::Vector{Int}, y::Vector{Int}, img)
        newimg = zeros(eltype(img), K, K)
        # flatten the image for efficient indexing
        img_flat = vec(img)
        # new linear indices
        new_indices = (y .- 1) * K + x
        newimg[new_indices] .= img_flat
        return newimg
end
function normalize(coordinates::Matrix{Float64})
        x_values = coordinates[:, 1]
        y_values = coordinates[:, 2]
        min_x = minimum(x_values)
        min_y = minimum(y_values)
        shift_x = min_x < 0 ? -min_x : 0
        shift_y = min_y < 0 ? -min_y : 0
        return hcat(x_values .+ shift_x, y_values .+ shift_y)
end

function snap_coords_parallel(coords::Matrix{Float64}, K::Int)
        x_values = coords[:, 1]
        y_values = coords[:, 2]
        Lx = sortperm(x_values)
        ranks = zeros(Int, K^2)
        @inbounds ranks[Lx] .= 1:K^2  # array of ranks of each x
        x_new = div.(ranks .- 1, K) .+ 1  # Equivalent to floor((rank-1)/K) + 1
        sorted_indices = sortperm(collect(1:K^2), by=i -> (x_new[i], y_values[i]))
        sorted_x_new = x_new[sorted_indices]
        y_new = zeros(Int, K^2)
        # Identify group boundaries where x_new changes
        change = [sorted_x_new[j] != sorted_x_new[j+1] for j in 1:((K^2)-1)]
        group_start = [1; findall(change .== true) .+ 1]
        group_end = [findall(change .== true); K^2]
        num_groups = length(group_start)
        # Parallel assignment of y_new
        @threads for g in 1:num_groups
                start_idx = group_start[g]
                end_idx = group_end[g]
                group_size = end_idx - start_idx + 1
                for j in 1:group_size
                        idx = sorted_indices[start_idx+j-1]
                        y_new[idx] = j
                end
        end
        return x_new, y_new
end

function snap_coords(coords::Matrix{Float64}, K::Int)
        x_values = coords[:, 1]
        y_values = coords[:, 2]
        Lx = sortperm(x_values)
        ranks = zeros(Int, K^2)
        @inbounds ranks[Lx] .= 1:K^2  # array of ranks of each x
        x_new = div.(ranks .- 1, K) .+ 1  # Equivalent to floor((rank-1)/K) + 1
        sorted_indices = sortperm(collect(1:K^2), by=i -> (x_new[i], y_values[i]))
        y_new = zeros(Int, K^2)
        # Identify group boundaries where x_new changes
        current_x = 0
        current_rank = 0
        for idx in sorted_indices
                if x_new[idx] != current_x
                        current_x = x_new[idx]
                        current_rank = 1
                else
                        current_rank += 1
                end
                y_new[idx] = current_rank
        end
        return x_new, y_new
end

# compute coord_transform using vectorization
function net_transform(x::Vector{Float64}, y::Vector{Float64}, n::Int, K::Int)
        N = K^2
        # Reshape into n rows
        x_matrix = reshape(x, n, :)
        y_matrix = reshape(y, n, :)

        # Sum along the first dimension (rows)
        x_summed = sum(x_matrix, dims=1)
        y_summed = sum(y_matrix, dims=1)

        # Combine into coord_transform
        coord_transform = hcat(vec(x_summed), vec(y_summed))
        return coord_transform
end

function compute_coord_transform_parallel(x::Vector{Float64}, y::Vector{Float64}, n::Int, K::Int)
        N = K^2
        pad_length = n - (N % n)
        if pad_length != n
                x_padded = vcat(x, zeros(Float64, pad_length))
                y_padded = vcat(y, zeros(Float64, pad_length))
        else
                x_padded = x
                y_padded = y
        end

        total_chunks = div(length(x_padded), n)
        coord_transform = Vector{Tuple{Float64,Float64}}(undef, total_chunks)

        @threads for chunk in 1:total_chunks
                start_idx = (chunk - 1) * n + 1
                end_idx = chunk * n
                sum_x = sum(x_padded[start_idx:end_idx])
                sum_y = sum(y_padded[start_idx:end_idx])
                coord_transform[chunk] = (sum_x, sum_y)
        end

        return coord_transform
end

function tweak_inits(hash, x0, y0, k, u)
        key1, key2 = parse(BigInt, hash[1:32]; base=16), parse(BigInt, hash[33:64]; base=16)
        key3 = xor(key1, key2)
        key3 = lpad(string(key3, base=16), 32, '0')
        # split key3 into 32 group of ints
        H = [parse(Int, h; base=16) for h in collect(key3)]

        # get epsilons to transform inital conditons
        eps = [reduce(xor, H[i:i+7]) for i in 1:8:32] ./ 15
        eta = (x0 / y0) * (u / k)
        #get new initial conditions
        new_x0 = mod((eps[1] + eta) / (x0 + eta), 1)
        new_y0 = mod((eps[2] + eta) / (y0 + eta), 1)
        new_u = u + eps[3] / eta
        new_k = k + eps[4] / eta
        return new_x0, new_y0, new_u, new_k

end

function brownian_motion(x_coords, y_coords, n, K)
        r = mod.(Int.(floor.((x_coords + y_coords) .* 10^8)), 101)
        az = mod.(Int.(floor.(y_coords .* 10^8)), 361)
        theta = mod.(Int.(floor.(x_coords .* 10^8)), 181)
        # calclate corresponding cartesian motion
        x = r .* sin.(theta) .* cos.(az)
        y = r .* sin.(theta) .* sin.(az)

        # make an empty coordinate grid
        i_rep = repeat(1:K, outer=K)
        j_rep = repeat(1:K, inner=K)
        coord_grid = hcat(Float64.(i_rep), Float64.(j_rep))

        # get the final coordinates of each pixel after net motion
        coords = net_transform(x, y, n, K) .+ coord_grid # + float.(vec([[i, j] for i in 1:K, j in 1:K]))
        coords = normalize(coords) # to positive xy plane


end

function chaos_matrices(K, X2, X22, Y2, Y22)
        A1 = UInt8.(reshape(mod.(Int.(floor.((X2 + Y2 .+ 1) .* 10^8)), 16), K, K))
        A2 = UInt8.(reshape(mod.(Int.(floor.((X22 + Y22 .+ 1) .* 10^8)), 16), K, K))
        return A1, A2
end

function extract_high_nibble(byte::UInt8)::UInt8
        return (byte >> 4) & 0x0F
end

function extract_low_nibble(byte::UInt8)::UInt8
        return byte & 0x0F
end

function combine_nibbles(a::UInt8, b::UInt8)::UInt8
        # Ensure 'a' and 'b' are within 0-15
        a_masked = a & 0x0F
        b_masked = b & 0x0F

        # Shift 'a' to the higher nibble and combine with 'b'
        combined = (a_masked << 4) | b_masked

        return combined
end

function diffuse(k, new_k, K, A1, A2, Q2h, Q2l, Q3hd, Q3ld)
        for i in 1:K, j in 1:K
                if j != 1
                        Q3ld[i, j] = xor(Q2l[i, j], A1[i, j], A2[K+1-i, K+1-j], UInt8(mod(Int(floor((1 - 1.4 * (Q3ld[i, j-1] / 15.0)^2 + (Q3hd[i, j-1] / 15.0)) * 10^8)), 16)))
                        Q3hd[i, j] = xor(Q2h[i, j], A1[K+1-i, K+1-j], A2[i, j], UInt8(mod(Int(floor(0.3 * (Q3ld[i, j-1] / 15.0) * 10^8)), 16)))
                elseif j == 1 && i != 1
                        Q3ld[i, j] = xor(Q2l[i, j], A1[i, j], A2[K+1-i, K+1-j], UInt8(mod(Int(floor((1 - 1.4 * (Q3ld[i-1, K] / 15.0)^2 + (Q3hd[i-1, K] / 15.0)) * 10^8)), 16)))
                        Q3hd[i, j] = xor(Q2h[i, j], A1[K+1-i, K+1-j], A2[i, j], UInt8(mod(Int(floor(0.3 * (Q3hd[i-1, K] / 15.0) * 10^8)), 16)))
                elseif j == 1 && i == 1
                        Q3ld[i, j] = xor(Q2l[i, j], A1[i, j], A2[K+1-i, K+1-j], UInt8(mod(Int(floor((1 - 1.4 * (k / 15.0)^2 + (new_k / 15.0)) * 10^8)), 16)))
                        Q3hd[i, j] = xor(Q2h[i, j], A1[K+1-i, K+1-j], A2[i, j], UInt8(mod(Int(floor(0.3 * (k / 15.0) * 10^8)), 16)))
                end

        end
        return combine_nibbles.(Q3hd, Q3ld)
end

## inverse of diffuse()
function refuse(k, new_k, K, A1, A2, Q2h, Q2l, Q3hd, Q3ld)
        for i in 1:K, j in 1:K
                if j != 1
                        Q2l[i, j] = xor(Q3ld[i, j], A1[i, j], A2[K+1-i, K+1-j], UInt8(mod(Int(floor((1 - 1.4 * (Q3ld[i, j-1] / 15.0)^2 + (Q3hd[i, j-1] / 15.0)) * 10^8)), 16)))
                        Q2h[i, j] = xor(Q3hd[i, j], A1[K+1-i, K+1-j], A2[i, j], UInt8(mod(Int(floor(0.3 * (Q3ld[i, j-1] / 15.0) * 10^8)), 16)))
                elseif j == 1 && i != 1
                        Q2l[i, j] = xor(Q3ld[i, j], A1[i, j], A2[K+1-i, K+1-j], UInt8(mod(Int(floor((1 - 1.4 * (Q3ld[i-1, K] / 15.0)^2 + (Q3hd[i-1, K] / 15.0)) * 10^8)), 16)))
                        Q2h[i, j] = xor(Q3hd[i, j], A1[K+1-i, K+1-j], A2[i, j], UInt8(mod(Int(floor(0.3 * (Q3hd[i-1, K] / 15.0) * 10^8)), 16)))
                elseif j == 1 && i == 1
                        Q2l[i, j] = xor(Q3ld[i, j], A1[i, j], A2[K+1-i, K+1-j], UInt8(mod(Int(floor((1 - 1.4 * (k / 15.0)^2 + (new_k / 15.0)) * 10^8)), 16)))
                        Q2h[i, j] = xor(Q3hd[i, j], A1[K+1-i, K+1-j], A2[i, j], UInt8(mod(Int(floor(0.3 * (k / 15.0) * 10^8)), 16)))
                end

        end
        return combine_nibbles.(Q2h, Q2l)
end

function start_encrypt()
        n = 20
        println("Enter x0 between 0 and 1:")
        x0 = parse(Float64, readline())
        println("Enter y0 between 0 and 1:")
        y0 = parse(Float64, readline())
        println("Enter u0:")
        u0 = parse(Float64, readline())
        println("Enter k, for best performance, choose k < 10. k> 40 will cause convergence:")
        k0 = parse(Float64, readline())
        println("Enter image path:")
        path = readline()
        # println("Enter output path for encrypted image:")
        # outpath = readline()
        # println("Enter output path for secrete key:")
        # keypath = readline()
        println("loading image")
        img = expand_img(Gray.(load(path)))
        out_img, key = main_encrypt(img, n, x0, y0, u0, k0)
        println("saving...")
        save(dirname(path) * "/encrypted_" * basename(path) * ".png", out_img)
        # TODO: replace with Binary Object Storage
        open(splitext(basename(path))[1] * "_key.json", "w") do io
                JSON.print(io, key)
        end
        println("done!")
end

function start_decrypt()
        println("Enter encrypted image path:")
        path = readline()
        println("Enter path to key:")
        key_path = readline()
        n = 20

        # path = "images/encrypted_horse.jpg"
        # key_path = "horse_key.json"
        # TODO: replace with Binary Object Storage
        key = JSON.parsefile(key_path)
        println("loading image")
        img = expand_img(Gray.(load(path)))
        Q1 = main_decrypt(img, key, n)
        println("saving...")
        save(dirname(path) * "/decrypted_" * basename(path), Q1)
        println("done!")
end

function allbench()
        Kt = 4000
        nt = 20
        Nt = 735^2
        rt = rand(Float64, Nt)
        thetat = rand(Float64, Nt)
        azt = rand(Float64, Nt)
        xt = rand(Kt^2)
        yt = rand(Kt^2)
        x0t = 0.32
        y0t = 0.2
        u0t = 6.0
        k0t = 5
        img = expand_img(Gray.(load("images/horse.jpg")))
        Q3, key = main_encrypt(img, nt, x0t, y0t, u0t, k0t)
        key1 = deepcopy(key)
        Q1 = main_decrypt(Q3, key1, nt)
end

function start()
        println("Welcome To the ChaoticEncrypt, an image encryption algorithm based on Chaos!")
        println("Would you like to Encrypt or Decrypt an image? (e/d):")
        choice = readline()
        if lowercase(choice) == "e" || lowercase(choice) == "encrypt"
                start_encrypt()
        elseif lowercase(choice) == "d" || lowercase(choice) == "decrypt"
                start_decrypt()
        else
                println("Please try again and enter a valid option (e or d)")
        end
end


function bench()
        img = expand_img(Gray.(load("images/mount.jpg")))
        println(size(img, 1))
        Q3, key = main_encrypt(img, 20, 0.32, 0.2, 6.0, 5)

        bench_enc = @benchmark main_encrypt(expand_img(Gray.(load("images/mount.jpg"))), 20, 0.32, 0.2, 6.0, 5)
        emin_time = minimum(bench_enc)
        emeann_time = mean(bench_enc)
        emem_alloc = bench_enc.memory

        key1 = deepcopy(key)
        bench_denc = @benchmark main_decrypt($Q3, $key1, 20)
        dmin_time = minimum(bench_denc)
        dmeann_time = mean(bench_denc)
        dmem_alloc = bench_enc.memory

        # bench_both = @benchmark allbench()
        # bmin_time = minimum(bench_both)
        # bmeann_time = mean(bench_both)
        # bmem_alloc = bench_both.memory

        println("ENCRYPTION")
        println("Minimum time: ", emin_time, " ns")
        println("Mean time: ", emeann_time, " ns")
        println("Memory allocated: ", emem_alloc / 1024^2, " MegaBytes")
        println("DECRYPTION")
        println("Minimum time: ", dmin_time, " ns")
        println("Mean time: ", dmeann_time, " ns")
        println("Memory allocated: ", dmem_alloc / 1024^2, " MegaBytes")
        # println("BOTH")
        # println("Minimum time: ", bmin_time, " ns")
        # println("Mean time: ", bmeann_time, " ns")
        # println("Memory allocated: ", bmem_alloc / 1024^2, " MegaBytes")
        #
end
function test()
        # Example Usage
        Kt = 4000
        nt = 20
        Nt = 735^2
        rt = rand(Float64, Nt)
        thetat = rand(Float64, Nt)
        azt = rand(Float64, Nt)
        xt = rand(Kt^2)
        yt = rand(Kt^2)
        x0t = 0.32
        y0t = 0.2
        u0t = 10.0
        k0t = 10
        img = expand_img(Gray.(load("images/horse.jpg")))
        Q3, key = main_encrypt(img, nt, x0t, y0t, u0t, k0t)
        open("testkey.json", "w") do io
                JSON.print(io, key)
        end
        save("encrypted_test.png", Q3)
        img2 = expand_img(Gray.(load("encrypted_test.png")))
        println(img2 == Q3)
        key1 = JSON.parsefile("testkey.json")
        Q1 = main_decrypt(img2, key1, nt)
        println(img == Q1)
        return Q3, Q1

end

start()

