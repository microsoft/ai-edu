// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.MiddlewareService
{
    public class LandmarkResult
    {
        public string Name { get; set; }

        public double Confidence { get; set; }

        public string Url { get; set; }

        public string Description { get; set; }

        public string ThumbnailUrl { get; set; }
    }
}