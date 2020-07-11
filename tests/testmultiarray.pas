unit testmultiarray;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, multiarray, numerik;

type

  TTestMultiArray = class(TTestCase)
  published
    procedure TestBroadcastMatrixScalar;
    procedure TestTranspose;
  end;

implementation

procedure TTestMultiArray.TestBroadcastMatrixScalar;
var
  M: TMultiArray;
begin
  M := TMultiArray([1, 2, 3, 4]).Reshape([2, 2]);
  AssertTrue(VectorsEqual((M + 2).GetVirtualData, [3, 4, 5, 6]));
end;

procedure TTestMultiArray.TestTranspose;
var
  M: TMultiArray;
begin
  M := [1, 2, 3, 4, 5, 6];
  M := M.Reshape([2, 3]);
  AssertTrue(VectorsEqual(M.T.GetVirtualData, [1, 4, 2, 5, 3, 6]));
end;

initialization
  RegisterTest(TTestMultiArray);

end.


