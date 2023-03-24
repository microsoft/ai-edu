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
    public class LandmarkService : ILandmarkService
    {
        private readonly IVisionService visionService;
        private readonly IEntitySearchService entityService;

        public LandmarkService(IVisionService vs, IEntitySearchService ess)
        {
            this.visionService = vs;
            this.entityService = ess;
        }

        public async Task<LandmarkResult> Do(byte[] imgData)
        {
            // get original recognized result
            var streamLandmark = Helper.GetStream(imgData);
            Landmark landmark = await this.visionService.RecognizeLandmarkAsync(streamLandmark);
            if (landmark != null)
            {
                // get entity search result
                string entityName = landmark.name;
                string jsonResult = await this.entityService.SearchEntityAsync(entityName);
                EntityResult er = JsonConvert.DeserializeObject<EntityResult>(jsonResult);
                // isolation layer: decouple data structure then return abstract result
                LandmarkResult lr = new LandmarkResult()
                {
                    Name = er.entities.value[0].name,
                    Description = er.entities.value[0].description,
                    Url = er.entities.value[0].url,
                    ThumbnailUrl = er.entities.value[0].image.thumbnailUrl,
                    Confidence = landmark.confidence
                };
                return lr;
            }
            return null;
        }
    }
}