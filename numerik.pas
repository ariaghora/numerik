unit numerik;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, math, multiarray;

  { Wrappers of single functions }
  function _Sin(X: Single): Single;
  function _DegToRad(X: Single): Single;

  function DegToRad(X: TMultiArray): TMultiArray; overload;
  function Sin(X: TMultiArray): TMultiArray; overload;

implementation

  function _Sin(X: Single): Single;
  begin
    Exit(Sin(X));
  end;

  function _DegToRad(X: Single): Single;
  begin
    Exit(DegToRad(X));
  end;

  function DegToRad(X: TMultiArray): TMultiArray; overload;
  begin
    Exit(ApplyUFunc(X, @_DegToRad));
  end;

  function Sin(X: TMultiArray): TMultiArray;
  begin
    Exit(ApplyUFunc(X, @_Sin));
  end;

end.

