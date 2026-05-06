using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace LlamaLauncher.Services
{
    public static class GgufService
    {
        public class ParsedResult
        {
            public string CurrentFilePath { get; set; } = "";
            public string DetectedArch { get; set; } = "";
            public string DetectedModelName { get; set; } = "";
            public uint NEmbd { get; set; }
            public uint NLayer { get; set; }
            public double EstimatedVramGb { get; set; }
            public double? ContextLength { get; set; }
            public long FileSizeBytes { get; set; }
            public int ShardCount { get; set; }
            public string FullMetaText { get; set; } = "";
            public string ChatTemplate { get; set; } = "";
            public List<TensorData> Tensors { get; set; } = new List<TensorData>();
            public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
        }

        public class TensorData
        {
            public string Name { get; set; } = "";
            public string Quant { get; set; } = "";
            public string Dimensions { get; set; } = "";
            public double SizeGb { get; set; }
            public string Offset { get; set; } = "";
        }

        public static ParsedResult Parse(string ggufPath)
        {
            // Calculate file size
            var fileInfo = new FileInfo(ggufPath);
            long fileSizeBytes = fileInfo.Length;

            using var fs = new FileStream(ggufPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var br = new BinaryReader(fs);

            if (br.ReadUInt32() != 0x46554747) throw new Exception("Not a GGUF file!");
            br.BaseStream.Seek(4, SeekOrigin.Current); // Skip version

            ulong tensorCount = br.ReadUInt64();
            ulong metadataCount = br.ReadUInt64();

            var metaDict = new Dictionary<string, object>();
            var sb = new StringBuilder();
            string chatTemp = "";
            string detectedArch = "UNKNOWN";
            uint n_embd = 0;
            uint n_layer = 0;
            double estimatedVram = 0;
            double? contextLength = null;

            // -------------------- METADATA --------------------
            for (ulong i = 0; i < metadataCount; i++)
            {
                string key = SafeReadString(br);
                uint type = br.ReadUInt32();
                object val = DecodeGGUFValue(br, type);

                metaDict[key] = val;

                if (key == "general.architecture")
                {
                    detectedArch = val?.ToString()?.ToLowerInvariant() ?? "UNKNOWN";
                }
                else if (key.Contains("embedding_length"))
                {
                    n_embd = ConvertToUInt32(val);
                }
                else if (key.Contains("block_count"))
                {
                    n_layer = ConvertToUInt32(val);
                }
                else if (key.Contains("context_length"))
                {
                    contextLength = ConvertToDouble(val);
                }

                if (key.Contains("chat_template"))
                {
                    chatTemp = val?.ToString() ?? "";
                }
                else if (!key.Contains("tokenizer.ggml.tokens") &&
                         !key.Contains("tokenizer.chat_template"))
                {
                    AppendFormattedLine(sb, key, val);
                }
            }

            // If no layer count found, estimate from file size
            if (n_layer == 0)
            {
                double sizeGb = fileSizeBytes / (1024.0 * 1024.0 * 1024.0);
                n_layer = sizeGb switch
                {
                    > 100 => 120,
                    > 50 => 80,
                    > 20 => 60,
                    > 10 => 40,
                    > 3 => 32,
                    _ => 24
                };
            }

            // -------------------- TENSORS --------------------
            var tensors = new List<TensorData>();
            for (ulong i = 0; i < tensorCount; i++)
            {
                string name = SafeReadString(br);
                uint nDims = br.ReadUInt32();

                ulong[] dims = new ulong[nDims];
                ulong totalElements = 1;
                for (uint d = 0; d < nDims; d++)
                {
                    dims[d] = br.ReadUInt64();
                    totalElements *= dims[d];
                }

                uint tType = br.ReadUInt32();
                ulong offset = br.ReadUInt64();

                double sizeGb = CalculateTensorSize(totalElements, tType);
                estimatedVram += sizeGb;

                tensors.Add(new TensorData
                {
                    Name = name,
                    Quant = GetQuantName(tType),
                    Dimensions = string.Join("x", dims),
                    SizeGb = sizeGb,
                    Offset = offset.ToString("X")
                });
            }

            return new ParsedResult
            {
                CurrentFilePath = ggufPath,
                DetectedArch = detectedArch,
                DetectedModelName = metaDict.TryGetValue("general.name", out var nameVal) ? nameVal.ToString() ?? "" : "",
                NEmbd = n_embd,
                NLayer = n_layer,
                ContextLength = contextLength,
                EstimatedVramGb = estimatedVram,
                FileSizeBytes = fileSizeBytes,
                ShardCount = 1,
                FullMetaText = sb.ToString(),
                ChatTemplate = chatTemp,
                Tensors = tensors,
                Metadata = metaDict
            };
        }

        // -------------------------------------------------
        //  HELPER METHODS
        // -------------------------------------------------
        private static string SafeReadString(BinaryReader br)
        {
            ulong len = br.ReadUInt64();
            if (len == 0 || len > 2_000_000) return "";
            return Encoding.UTF8.GetString(br.ReadBytes((int)len));
        }

        private static object DecodeGGUFValue(BinaryReader br, uint type)
        {
            switch (type)
            {
                case 0: return br.ReadByte();
                case 1: return br.ReadSByte();
                case 2: return br.ReadUInt16();
                case 3: return br.ReadInt16();
                case 4: return br.ReadUInt32();
                case 5: return br.ReadInt32();
                case 6: return br.ReadSingle();
                case 7: return br.ReadByte() != 0;
                case 8: return SafeReadString(br);
                case 9: // Array
                    return DecodeGGUFArray(br);
                case 10: return br.ReadUInt64();
                case 11: return br.ReadInt64();
                case 12: return br.ReadDouble();
                default:
                    br.BaseStream.Seek(GetTypeSize(type), SeekOrigin.Current);
                    return $"[UNKNOWN_TYPE:{type}]";
            }
        }

        private static object DecodeGGUFArray(BinaryReader br)
        {
            uint arrayType = br.ReadUInt32();
            ulong count = br.ReadUInt64();

            if (count == 0) return new string[0];

            if (arrayType == 8)
            {
                var strings = new List<string>();
                for (ulong i = 0; i < count; i++)
                {
                    strings.Add(SafeReadString(br));
                }
                return strings.ToArray();
            }

            long bytesToSkip = (long)count * GetTypeSize(arrayType);
            br.BaseStream.Seek(bytesToSkip, SeekOrigin.Current);
            return $"[Array:{count}]";
        }

        private static long GetTypeSize(uint type)
        {
            return type switch
            {
                0 or 1 or 7 => 1,    // UINT8, INT8, BOOL
                2 or 3 => 2,         // UINT16, INT16
                4 or 5 or 6 => 4,    // UINT32, INT32, FLOAT32
                10 or 11 or 12 => 8, // UINT64, INT64, FLOAT64
                _ => 4               // Default
            };
        }

        private static double CalculateTensorSize(ulong elements, uint type)
        {
            double bpw = type switch
            {
                0 => 32,   // F32
                1 => 16,   // F16
                2 => 5,    // Q4_0
                3 => 5,    // Q4_1
                7 => 6.5,  // Q5_K
                8 => 8.5,  // Q8_0
                12 => 4.5, // Q4_K
                14 => 6.5, // Q6_K
                _ => 8     // Default
            };
            return (elements * bpw / 8.0) / (1024.0 * 1024.0 * 1024.0);
        }

        private static string GetQuantName(uint t) => t switch
        {
            12 => "Q4_K",
            14 => "Q6_K",
            8 => "Q8_0",
            7 => "Q5_K",
            2 => "Q4_0",
            1 => "F16",
            0 => "F32",
            _ => $"T:{t}"
        };

        private static void AppendFormattedLine(StringBuilder sb, string key, object value)
        {
            string line = string.Format("{0,-40} : {1}", key, FormatValue(value));
            sb.AppendLine(line);
        }

        private static string FormatValue(object value)
        {
            if (value == null) return "null";

            if (value is Array array)
            {
                if (array.Length == 0) return "[]";
                if (array.Length > 5) return $"[Array of {array.Length} elements]";
                return $"[{string.Join(", ", array.Cast<object>().Take(5))}]";
            }

            return value.ToString();
        }

        private static uint ConvertToUInt32(object value)
        {
            try
            {
                return value switch
                {
                    byte b => b,
                    sbyte sb => (uint)sb,
                    ushort us => us,
                    short s => (uint)s,
                    uint ui => ui,
                    int i => (uint)i,
                    ulong ul => (uint)Math.Min(ul, uint.MaxValue),
                    long l => (uint)Math.Min(l, uint.MaxValue),
                    string s when uint.TryParse(s, out uint result) => result,
                    _ => 0
                };
            }
            catch
            {
                return 0;
            }
        }

        private static double? ConvertToDouble(object value)
        {
            try
            {
                return value switch
                {
                    byte b => (double)b,
                    sbyte sb => (double)sb,
                    ushort us => (double)us,
                    short s => (double)s,
                    uint ui => (double)ui,
                    int i => (double)i,
                    ulong ul => (double)ul,
                    long l => (double)l,
                    float f => (double)f,
                    double d => d,
                    string s when double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out double parsed) => parsed,
                    _ => null
                };
            }
            catch
            {
                return null;
            }
        }
    }
}
