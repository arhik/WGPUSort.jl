using Revise
using WGPUCompute
using WGPUCore 
using MacroTools
using WGSLTypes
using WGSLTypes: @letvar
WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

x = WgpuArray{Float32}(rand(Float32, 16, 16) .- 0.5f0)

@wgpukernel workgroupSizes=(4, 4) workgroupCount=(2, 2) function prefixsum_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	dim = UInt32(16)
	gId = dim*gIdy + gIdx
	out[gId] = Float32(gId)
end

function prefixsum(x)
	y = similar(x)
	prefixsum_kernel(x, y)
	return y
end

z = prefixsum(x)

