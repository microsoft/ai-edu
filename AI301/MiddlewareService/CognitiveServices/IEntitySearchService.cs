using System.Threading.Tasks;

namespace CognitiveMiddlewareService.CognitiveServices
{
    public interface IEntitySearchService
    {
        Task<string> SearchEntityAsync(string query);
    }
}
