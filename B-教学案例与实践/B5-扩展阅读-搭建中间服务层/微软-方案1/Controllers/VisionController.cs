// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using CognitiveMiddlewareService.CognitiveServices;
using CognitiveMiddlewareService.Processors;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;

namespace CognitiveMiddlewareService.Controllers
{
    [Route("api/[controller]")]
    public class VisionController : Controller
    {

        private readonly IProcessService processor;

        public VisionController(IProcessService ps)
        {
            this.processor = ps;
        }

        // GET api/values
        [HttpGet]
        public IEnumerable<string> Get()
        {
            return new string[] { "value1", "value2" };
        }

        // GET api/values/5
        [HttpGet("{id}")]
        public string Get(int id)
        {
            return "value";
        }

        // POST api/values
        [HttpPost]
        public async Task<string> Post([FromForm] IFormCollection formCollection)
        {
            try
            {
                IFormCollection form = await this.Request.ReadFormAsync();
                IFormFile file = form.Files.First();

                var bufferData = Helper.GetBuffer(file);
                var result = await this.processor.Process(bufferData);
                string jsonResult = JsonConvert.SerializeObject(result);
                // return json formatted data
                return jsonResult;
            }
            catch (Exception ex)
            {
                Debug.Write(ex.Message);
                return null;
            }
        }
    }
}