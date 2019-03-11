// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using NSwag;
using NSwag.CodeGeneration.CSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GenerationClientCode
{
    class Program
    {
        static void Main(string[] args)
        {
            Generate().Wait();
        }

        static async Task Generate()
        {
            var document = await SwaggerDocument.FromUrlAsync("https://goldennumber.aiedu.msra.cn/swagger/v1%20-%20English/swagger.json");

            var settings = new SwaggerToCSharpClientGeneratorSettings
            {
                ClassName = "GoldenNumberService",
                CSharpGeneratorSettings =
                {
                    Namespace = "GoldenNumber"
                }
            };

            var generator = new SwaggerToCSharpClientGenerator(document, settings);
            var code = generator.GenerateFile();

            var copyright = "// Copyright (c) Microsoft. All rights reserved.\r\n// Licensed under the MIT license. See LICENSE file in the project root for full license information.\r\n\r\n";

            File.WriteAllText("GoldenNumberService.cs", copyright + code.Replace("\r\n", "\n").Replace("\n", "\r\n"));
        }
    }
}
