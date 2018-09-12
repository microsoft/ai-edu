using CognitiveMiddlewareService.CognitiveServices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.MiddlewareService
{
    public interface ICelebrityService
    {
        Task<CelebrityResult> Do(byte[] imgData);
    }
}
