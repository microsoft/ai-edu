// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Threading.Tasks;

namespace CognitiveMiddlewareService.CognitiveServices
{
    public interface IEntitySearchService
    {
        Task<string> SearchEntityAsync(string query);
    }
}
