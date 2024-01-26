using Revise
using WGPUCompute
using WGPUCore 
using MacroTools
using WGSLTypes
using WGSLTypes: @letvar
WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

x = WgpuArray{Float32}(rand(Float32, 8, 8) .- 0.5f0)

@wgpukernel workgroupSizes=(4, 4) workgroupCount=(2, 2) function prefixsum_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = T(gId)
end

@wgpukernel workgroupSizes=(4, 4) workgroupCount=(2, 2) function cast_kernel(x::WgpuArray{T, N}, out::WgpuArray{S, N}) where {T, S, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = S(ceil(x[gId]))
end

function cast(S::DataType, x::WgpuArray{T, N}) where {T, N}
	y = WgpuArray{S}(undef, size(x))
	cast_kernel(x, y)
	return y
end

function prefixsum(x)
	y = similar(x)
	prefixsum_kernel(x, y)
	return y
end

z = prefixsum(x)
z = cast(UInt32, x)

# TODO
# y = cast(Bool, x)
