// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using CognitiveMiddlewareService.CognitiveServices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.MiddlewareService
{
    public interface ILandmarkService
    {
        Task<LandmarkResult> Do(byte[] imgData);
    }
}
