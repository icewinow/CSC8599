#version 330 core

out float FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D texNoise;

uniform mat4 projection;
uniform mat4 unProjection;

uniform float RadiusToScreen;

// Parameters
uniform int directions;
uniform int steps;
uniform float bias;
uniform float radius;
uniform float NegInvR2;
uniform vec2 InvResolutionDirection;
uniform vec2 noiseScale;
uniform float AOMultiplier;

const float PI = 3.14159265359;

//----------------------------------------------------------------------------------
vec3 FetchViewPos(vec2 UV)
{
  vec3 viewPos = texture(gPosition, UV.xy).xyz;
  return viewPos;

}

//----------------------------------------------------------------------------------
float Falloff(float DistanceSquare){
  return DistanceSquare * NegInvR2 + 1.0;
}


//----------------------------------------------------------------------------------
vec2 RotateDirection(vec2 Dir, vec2 CosSin)
{
  return vec2(Dir.x*CosSin.x - Dir.y*CosSin.y,
              Dir.x*CosSin.y + Dir.y*CosSin.x);
}

//----------------------------------------------------------------------------------
// P = view-space position at the kernel center
// N = view-space normal at the kernel center
// S = view-space position of the current sample
//----------------------------------------------------------------------------------
float ComputeAO(vec3 P, vec3 N,vec3 S){
vec3 H = S - P;
float HdotH = dot(H, H); // compute length
float NdotH = dot(N, H) * 1.0/sqrt(HdotH);

return clamp(NdotH - bias,0,1) * clamp(Falloff(HdotH),0,1);

}


//----------------------------------------------------------------------------------
float ComputeCoarseAO(vec2 fragUV, float RadiusToScreen , vec3 randomVec, vec3 ViewPosition, vec3 ViewNormal){
  // Divide by step + 1 so that the farthest samples are not fully attenuated
  float StepSizePixels = RadiusToScreen  / (steps + 1);
  
  float Alpha = 2.0 * PI / directions;
  float AO = 0;
  
  for(float DirectionIndex  = 0; DirectionIndex < directions; ++DirectionIndex)
  {
    float Angle = Alpha * DirectionIndex;

    // Compute normalized 2D direction
    vec2 Direction = RotateDirection(vec2(cos(Angle), sin(Angle)), randomVec.xy);

    // Jitter starting sample within the first step
    float RayPixels = (randomVec.z * StepSizePixels);

    for(float StepIndex = 0; StepIndex < steps; ++StepIndex)
    {
       vec2 SnappedUV = round(RayPixels * Direction) * InvResolutionDirection + fragUV;
       vec3 S = FetchViewPos(SnappedUV);

       RayPixels += StepSizePixels;
       
       AO += ComputeAO(ViewPosition,ViewNormal,S);
    }

  }

  AO *= AOMultiplier / (steps * directions );
  return clamp(1.0 - AO * 2.0,0,1);

}

void main (void) {

// get input for HBAO algorithm
  vec3 fragPos = texture(gPosition, TexCoords).xyz;
  vec3 normal = normalize(texture(gNormal, TexCoords).rgb);
  vec3 randomVec = normalize(texture(texNoise, TexCoords * noiseScale).xyz);

//UV of kernel center
  vec4 fragUV = vec4(fragPos,1.0);
  fragUV = projection * fragUV;
  fragUV.xyz /= fragUV.w; //(-1,1)
  fragUV.xyz = fragUV.xyz * 0.5 + 0.5;

  float AO = ComputeCoarseAO(fragUV.xy, RadiusToScreen, randomVec, fragPos, normal);

  FragColor = AO;
}