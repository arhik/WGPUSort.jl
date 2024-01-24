using Revise
using WGPUCompute
using WGPUCore 
using MacroTools

WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

x = WgpuArray{Float32}(rand(Float32, 32, 32) .- 0.5f0)

@wgpukernel workgroupSizes=(16, 16) workgroupCount=(4,) function prefixsum_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.y * numWorkGroups.x + globalId.x
	out[gIdx] = Float32(gIdx)
end

@wgpukernel workgroupSizes=(16, 16) workgroupCount=(4,) function radixSort_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.y * numWorkGroups.x + globalId.x
	out[gIdx] = Float32(gIdx)
end

function radixSort(x)
	y = similar(x)
	radixSort_kernel(x, y)
	return y
end

function prefixsum(x)
	y = similar(x)
	prefixsum_kernel(x, y)
	return y
end

y = radixSort(x)
z = prefixsum(x)
