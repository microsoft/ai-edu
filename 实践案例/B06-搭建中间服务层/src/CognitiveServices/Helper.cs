// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Microsoft.AspNetCore.Http;
using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.CognitiveServices
{
    public class Helper
    {
        public static byte[] GetBuffer(IFormFile formFile)
        {
            Stream stream = formFile.OpenReadStream();
            MemoryStream memoryStream = new MemoryStream();
            formFile.CopyTo(memoryStream);
            var buffer = memoryStream.GetBuffer();
            return buffer;
        }

        public static MemoryStream GetStream(byte[] buffer)
        {
            if (buffer == null)
            {
                return null;
            }

            return new MemoryStream(buffer, false);
        }

        public static async Task<string> MakeGetRequest(HttpClient httpClient, string uri, string key)
        {
            try
            {
                using (var request = new HttpRequestMessage(HttpMethod.Get, uri))
                {
                    request.Headers.Add("Ocp-Apim-Subscription-Key", key);
                    using (HttpResponseMessage response = await httpClient.SendAsync(request))
                    {
                        if (response.IsSuccessStatusCode)
                        {
                            string jsonResult = await response.Content.ReadAsStringAsync();
                            return jsonResult;
                        }
                    }
                }
                return null;
            }
            catch (Exception ex)
            {
                return null;
            }
        }
    }
}