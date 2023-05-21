using UnityEngine;
using UnityEngine.UI;
using ComputeUnits = MLStableDiffusion.ComputeUnits;

public sealed class Tester : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] RenderTexture _source = null;
    [SerializeField] RenderTexture _destination = null;
    [SerializeField] string _prompt = "sketch";
    [SerializeField, Range(0, 1)] float _strength = 0.5f;
    [SerializeField, Range(2, 10)] int _stepCount = 5;
    [SerializeField, Range(3, 15)] float _guidance = 7;

    #endregion

    #region Project asset references

    [SerializeField, HideInInspector] ComputeShader _preprocessShader = null;

    #endregion

    #region Stable Diffusion pipeline objects

    string ResourcePath
      => Application.streamingAssetsPath + "/StableDiffusion";

    MLStableDiffusion.ResourceInfo ResourceInfo
      => MLStableDiffusion.ResourceInfo.FixedSizeModel(ResourcePath, 640, 384);

    MLStableDiffusion.Pipeline _pipeline;

    #endregion

    #region Async operations


    #endregion

    #region MonoBehaviour implementation

    async void Start()
    {
        _pipeline = new MLStableDiffusion.Pipeline(_preprocessShader);
        await _pipeline.InitializeAsync(ResourceInfo, ComputeUnits.CpuAndGpu);

        while (true)
        {
            _pipeline.Prompt = _prompt;
            _pipeline.Strength = _strength;
            _pipeline.StepCount = _stepCount;
            _pipeline.Seed = Random.Range(0, 0x7fffffff);
            _pipeline.GuidanceScale = _guidance;
            await _pipeline.RunAsync(_source, _destination, destroyCancellationToken);
        }
    }

    void OnDestroy()
    {
        _pipeline?.Dispose();
        _pipeline = null;
    }

    #endregion
}
