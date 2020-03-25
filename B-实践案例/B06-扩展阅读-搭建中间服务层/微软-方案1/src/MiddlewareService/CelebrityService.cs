// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using CognitiveMiddlewareService.CognitiveServices;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.MiddlewareService
{
    public class CelebrityService : ICelebrityService
    {
        private readonly IVisionService visionService;
        private readonly IEntitySearchService entityService;

        public CelebrityService(IVisionService vs, IEntitySearchService ess)
        {
            this.visionService = vs;
            this.entityService = ess;
        }

        public async Task<CelebrityResult> Do(byte[] imgData)
        {
            // get original recognized result
            var stream = Helper.GetStream(imgData);
            Celebrity celebrity = await this.visionService.RecognizeCelebrityAsync(stream);
            if (celebrity != null)
            {
                // get entity search result
                string entityName = celebrity.name;
                string jsonResult = await this.entityService.SearchEntityAsync(entityName);
                EntityResult er = JsonConvert.DeserializeObject<EntityResult>(jsonResult);
                if (er?.entities?.value.Length > 0)
                {
                    // isolation layer: decouple data structure then return abstract result
                    CelebrityResult cr = new CelebrityResult()
                    {
                        Name = er.entities.value[0].name,
                        Description = er.entities.value[0].description,
                        Url = er.entities.value[0].url,
                        ThumbnailUrl = er.entities.value[0].image.thumbnailUrl,
                        Confidence = celebrity.confidence
                    };
                    return cr;
                }
            }
            return null;
        }
    }
}