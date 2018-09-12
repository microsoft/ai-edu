// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Threading.Tasks;

namespace CognitiveMiddlewareService.Processors
{
    public interface IProcessService
    {
        Task<AggregatedResult> Process(byte[] imgData);
    }
}
