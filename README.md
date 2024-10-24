# Chaotic ‘confusion - Rubik's cube transformation - diffusion’ image encryption algorithm
Working protoype /implementation of the black and white image encryption algorithm based on the 2D-LCCM (2D Logistic-Chebyshev chaotic map) proposed by Zhao et al. (Zhao et al., 2023, https://doi.org/10.1016/j.ijleo.2022.170342).

## Running the program
The implementation is written entirely in Julia and requires the follwing Libraries to be installed:
`Images, SHA, FileIO, ImageShow, Printf, ImageIO, JSON, BenchmarkTools`
1. Clone this repo
2. cd into ChaoticEncrypt
3. run `julia main.jl`

You will then be prompted to enter the path for the image to encrypt / decrypt. Unfortuantly this implementation can only output PNG images as lossy compresssion tampers with the image inforamtion leading to an imperfect decryption.
The encryption process will save the key in your current directory, and the encryted image in the same directory the image lives in.

