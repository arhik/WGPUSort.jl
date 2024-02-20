using Revise
using WGPUCompute
using WGPUCore 
using MacroTools
using WGSLTypes
using WGSLTypes: @letvar
using TracyProfiler_jll
using Tracy

WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

run(TracyProfiler_jll.tracy(); wait=false)

x = WgpuArray{Float32}(rand(Float32, 2048, 2048) .- 0.5f0)

@wgpukernel workgroupSizes=(16, 16) workgroupCount=(128, 128) function prefixsum_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = T(gId)
end

@wgpukernel workgroupSizes=(16, 16) workgroupCount=(128, 128) function cast_kernel(x::WgpuArray{T, N}, out::WgpuArray{S, N}) where {T, S, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = S(x[gId])
end

@tracepoint function cast(S::DataType, x::WgpuArray{T, N}) where {T, N}
	y = WgpuArray{S}(undef, size(x))
	cast_kernel(x, y)
	return y
end

@tracepoint "prefixCall" function prefixsum(x)
	@tracepoint "similar" y = similar(x)
	@tracepoint "prefixsum_kernel" prefixsum_kernel(x, y)
	return y
end

z = prefixsum(x)
z = cast(UInt32, x)

# TODO
# y = cast(Bool, x)
