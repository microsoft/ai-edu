// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.CognitiveServices
{
    public class VisionResult
    {
        public Result result { get; set; }
        public string requestId { get; set; }
    }

    public class Result
    {
        public Landmark[] landmarks { get; set; }
        public Celebrity[] celebrities { get; set; }
    }

    public class Landmark
    {
        public string name { get; set; }
        public double confidence { get; set; }
    }

    public class Celebrity
    {
        public virtual string name { get; set; }

        public virtual double confidence { get; set; }

    //    public FaceRectangle FaceRectangle { get; set; }
    }


}