using WGPUCompute
using WGPUCore 
using MacroTools

WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Debug)

x = WgpuArray{Float32}(undef, 8)

@kernel function radixSort(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end

(@macroexpand @kernel function radixSort(x::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end) |> MacroTools.striplines

radixSort(x)
