using System.Threading.Tasks;

namespace CognitiveMiddlewareService.Processors
{
    public interface IProcessService
    {
        Task<AggregatedResult> Process(byte[] imgData);
    }
}
